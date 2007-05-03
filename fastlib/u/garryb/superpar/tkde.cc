#include "fastlib/fastlib.h"

class Tkde {
 public:
  typedef DHrectBound Bound;
  typedef EpanKernel Kernel;
  
  struct TkdeParam {
    Kernel kernel;
    index_t dim;
    
    void Init(datanode *datanode) {
      kernel.Init(fx_param_double_req(datanode, "h"));
    }
    
    /**
     * Compute kernel sum for a region of reference points assuming we have the
     * actual query point.
     */
    double ComputeKernelSum(
        const Vector& q_point,
        index_t r_count, const Vector& r_mass, double r_sumsq) const {
      double quadratic_term = 
          + r_count * la::Dot(q_point, q_point)
          - 2.0 * la::Dot(q_point, r_mass)
          + r_sumsq;
      return r_count - quadratic_term * kernel.inv_bandwidth_sq();
    }

    static void ComputeCenter(
        index_t count, const Vector& mass, Vector* center) {
      center->Copy(mass);
      la::Scale(1.0 / count, center);
    }

    /**
     * Compute kernel sum given only a squared distance.
     */
    double ComputeKernelSum(
        double distance_squared,
        index_t r_count, const Vector& r_center, double r_sumsq) const {
      double quadratic_term =
          (distance_squared - la::Dot(r_center, r_center)) * r_count
          + r_sumsq;
      
      return r_count - quadratic_term * kernel.inv_bandwidth_sq();
    }
  };

  struct BlankInfo {
    template<typename Serializer>
    void Serialize(Serializer *s) const {}
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {}
  };

  typedef BlankInfo QInfo;
  typedef BlankInfo RInfo;

  struct MomentInfo {
    ALLOW_COPY(MomentInfo);
    
    Vector mass;
    double sumsq;
    index_t count;

    template<typename Serializer>
    void Serialize(Serializer *s) const {
      mass->Serialize(s);
      s->Put(sumsq);
      s->Put(count);
    }
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {
      mass->Deserialize(s);
      s->Get(&sumsq);
      s->Get(&count);
    }
    
    void Init(const TkdeParam& param) {
      mass.Init(param.dim);
      Reset();
    }
    
    void Reset() {
      mass.SetZero();
      sumsq = 0;
      count = 0;
    }
    
    void Add(index_t count_in, const Vector& mass_in, double sumsq_in) {
      if (unlikely(count_in != 0)) {
        la::AddTo(mass_in, &mass);
        sumsq += sumsq_in;
        count += count_in;
      }
    }
    
    void Add(const MomentInfo& other) {
      Add(other.count, other.mass, other.sumsq);
    }
    
    double ComputeKernelSum(const TkdeParam& param, const Vector& point) const {
      return param.ComputeKernelSum(point, count, mass, sumsq);
    }
    
    DRange ComputeKernelSumRange(const TkdeParam& param,
        const Bound& query_bound) const {
      DRange density_bound;
      Vector center;
      
      param.ComputeCenter(count, mass, &center);
      
      density_bound.lo = param.ComputeKernelSum(
          query_bound.MaxDistanceSqToPoint(center),
          count, center, sumsq);
      density_bound.hi = param.ComputeKernelSum(
          query_bound.MinDistanceSqToPoint(center),
          count, center, sumsq);
      
      return density_bound;
    }
    
    bool is_empty() const {
      return likely(count == 0);
    }
  };

  struct TkdeStat {
    MomentInfo moment_info;

    template<typename Serializer>
    void Serialize(Serializer *s) const {
      moment_info->Serialize(s);
    }
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {
      moment_info->Deserialize(s);
    }
    
    void InitZero(const TkdeParam& param) {
      moment_info.Init(param);
    }
    
    void Accumulate(const TkdeParam& param, const Vector& point,
        const RInfo& q_info) {
      moment_info.Add(1, point, la::Dot(point, point));
    }
    
    void Accumulate(const TkdeParam& param,
        const TkdeStat& stat, const Bound& bound, index_t n) {
      moment_info.Add(stat.moment_info);
    }
    
    void Postprocess(const TkdeParam& param, const Bound& bound, index_t n) {}
  };

  typedef BinarySpaceTree<Bound, Matrix, TkdeStat> RNode;
  typedef BinarySpaceTree<Bound, Matrix, EmptyStatistic<Matrix> > QNode;

  enum Label {
    LAB_LO = 2,
    LAB_UNKNOWN = 0,
    LAB_HI = 1,
    LAB_CONFLICT = 3
  };

  /**
   * Coarse result on a region.
   */
  struct TkdePostponed {
    /** Moments of pruned things. */
    MomentInfo moment_info;
    /** We pruned an entire part of the tree with a particular label. */
    Label label;
    
    template<typename Serializer>
    void Serialize(Serializer *s) const {
      moment_info.Serialize(s);
      s->Put(label);
    }
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {
      moment_info->Deserialize(s);
      s->Get(&label);
    }

    void Init(const TkdeParam& param) {
      moment_info.Init(param);
    }
    
    void Reset(const TkdeParam& param) {
      moment_info.Reset();
    }

    void ApplyPostponed(const TkdeParam& param, const TkdePostponed& other) {
      label |= other.label;
      DEBUG_ASSERT_MSG(label != LAB_CONFLICT, "Conflicting labels?");
      moment_info.Add(other.moment_info);
    }
  };

  /**
   * Coarse result on a region.
   */
  struct TkdeDelta {
    /** Density update to apply to children's bound. */
    DRange d_density;

    template<typename Serializer>
    void Serialize(Serializer *s) const {
      d_density.Serialize(s);
    }
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {
      d_density.Deserialize(s);
    }
    
    void Init(const TkdeParam& param) {
      d_density.Init(0, 0);
    }
    
    void ApplyDelta(const TkdeParam& param, const TkdeDelta& other) {
      d_density += other.d_density;
    }
  };

  // rho, but a bit of phi and lambda
  struct TkdeResult {
    double density;
    Label label;

    template<typename Serializer>
    void Serialize(Serializer *s) const {
      s->Put(density);
      s->Put(label);
    }
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {
      s->Get(&density);
      s->Get(&label);
    }

    void Init(const TkdeParam& param,
        const Vector& q_point, const QInfo& q_info,
        const RNode& r_root) {
      density.Init(0, 0);
      label = LAB_UNKNOWN;
    }

    void Postprocess(const TkdeParam& param,
        const Vector& q_point, const QInfo& q_info,
        const RNode& r_root) {
      /* nothing special to do */
    }

    void ApplyDelta(const TkdeParam& param,
        const TkdeDelta& delta) {
      density += delta.d_density;
    }

    void ApplyPostponed(const TkdeParam& param,
        const TkdePostponed& postponed,
        const Vector& q_point) {
      label |= postponed.label; /* bitwise OR */
      
      if (!postponed.moment_info.is_empty()) {
        density += postponed.moment_info.ComputeKernelSum(param, q_point);
      }
    }
  };

  class TkdeGlobalResult {
    template<typename Serializer>
    void Serialize(Serializer *s) const {}
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {}
    void Init(const TkdeParam& param) {}
    void Accumulate(const TkdeParam& param,
        const TkdeGlobalResult& other_global_result) {}
    void ApplyDelta(const TkdeParam& param, const TkdeDelta& delta) {}
    void UndoDelta(const TkdeParam& param, const TkdeDelta& delta) {}
    void Postprocess(const TkdeParam& param) {}
  };

  struct TkdeMassResult {
    /** Bound on density from leaves. */
    DRange density;
    Label label;

    template<typename Serializer>
    void Serialize(Serializer *s) const {
      density.Serialize(s);
      s->Put(label);
    }
    template<typename Deserializer>
    void Deserialize(Deserializer *s) {
      density.Deserialize(s);
      s->Get(&label);
    }

    void Copy(const TkdeMassResult& other) {
      density = other.density;
      label = other.label;
    }

    void Init(const TkdeParam& param) {
      /* horizontal init */
      density.Init(0, 0);
      label = 0;
    }

    void StartReaccumulate(const TkdeParam& param, const QNode& q_node) {
      /* vertical init */
      density.InitEmptySet();
      label = LAB_CONFLICT;
    }

    void Accumulate(const TkdeParam& param, const TkdeResult& result) {
      // TODO: applying to single result could be made part of Result,
      // but in some cases may require a copy/undo stage
      density |= result.density;
      label &= result.label;
    }

    void Accumulate(const TkdeParam& param,
        const TkdeMassResult& result, index_t n_points) {
      density |= result.density;
      density &= result.label;
    }

    void FinishReaccumulate(const TkdeParam& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }
    
    /** horizontal join operator */
    void ApplyMassResult(const TkdeParam& param,
        const TkdeMassResult& mass_result) {
      density += mass_result.density;
      label |= mass_result.label;
      DEBUG_ASSERT(label != LAB_CONFLICT);
    }

    void ApplyDelta(const TkdeParam& param,
        const TkdeDelta& delta) {
      density += delta.d_density;
    }
    
    bool ApplyPostponed(const TkdeParam& param,
        const TkdePostponed& postponed, const QNode& q_node) {
      bool change_made;
      
      if (unlikely(postponed.label)) {
        label = postponed.label;
        change_made = true;
      } else  if (unlikely(!postponed_moment_info.is_empty())) {
        density += moment_info.ComputeKernelSumRange(param, q_node.bound());
        change_made = true;
      } else {
        change_made = false;
      }
      
      return change_made;
    }
  };

  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct TkdeVectorPairVisitor {
    double density;
    
    void Init(const TkdeParam& param) {}
    
    bool StartVisitingQueryPoint(const TkdeParam& param,
        const Vector& q_point,
        const RNode& r_node,
        const TkdeMassResult& unapplied_mass_results,
        TkdeResult* q_result,
        TkdeGlobalResult* global_result) {
      if (unlikely(q_result->label != LAB_UNKNOWN)) {
        return false;
      }

      double distance_sq_lo = r_node.bound().MinDistanceSqToPoint(q_point);

      if (unlikely(distance_sq_lo > param.kernel.bandwidth_sq())) {
        return false;
      }

      double distance_sq_hi = r_node.bound().MaxDistanceSqToPoint(q_point);

      if (unlikely(distance_sq_hi < param.kernel.bandwidth_sq())) {
        q_result->density += r_node.stat().moment_info.ComputeKernelSum(
            param, q_point);
        return false;
      }
      
      density = 0;
      
      return true;
    }
    
    void VisitPair(const TkdeParam& param,
        const Vector& q_point, const QInfo& q_info,
        const Vector& r_point, const RInfo& r_info, index_t r_index) {
      double distance = la::DistanceSqEuclidean(q_point, r_point);
      density += param.kernel.EvalUnnormOnSq(distance);
    }
    
    void FinishVisitingQueryPoint(const TkdeParam& param,
        const Vector& q_point,
        const RNode& r_node,
        const TkdeMassResult& unapplied_mass_results,
        TkdeResult* q_result,
        TkdeGlobalResult* global_result) {
      q_result->density += density;

      double adjusted_threshold = param.threshold - q_result->density;
      
      if (unlikely(unapplied_mass_results->density.lo - EPS
          > adjusted_threshold)) {
        q_result->label = LAB_HI;
      } else if (unlikely(unapplied_mass_result->density.hi + EPS
          < adjusted_threshold)) {
        q_result->label = LAB_LO;
      }
    }
  };

  class TkdeAlgorithm {
    static bool ConsiderPairIntrinsic(
        const TkdeParam& param,
        const QNode& q_node,
        const RNode& r_node,
        TkdeDelta* delta,
        TkdeMassResult* q_mass_result,
        TkdeGlobalResult* global_result,
        TkdePostponed* q_postponed) {
      double distance_sq_lo =
          q_node.bound().MinDistanceSqToBound(r_node.bound());
      bool need_expansion;
      
      if (distance_sq_lo >= param.kernel.bandwidth_sq()) {
        delta->Init(0, 0);
        need_expansion = false;
      } else {
        double distance_sq_hi =
            q_node.bound().MaxDistanceSqToBound(r_node.bound());
        
        if (distance_sq_hi <= param.kernel.bandwidth_sq()) {
          q_postponed->moment_info.Add(r_node.stat().moment_info);
          delta->Init(0, 0);
          need_expansion = false;
        } else {
          delta->d_density = r_node.stat().moment_info.ComputeKernelSumRange(
              param, q_node.bound());
          // we computed the lower bound of the quadratic.  if it is positive
          // it means we have a better-than-nothing bound; if it is not, then
          // we can resort to saying the min contribution is zero.
          delta->d_density.lo = max(delta->d_density.lo, 0.0);
          q_mass_result.density += delta->d_density;
          need_expansion = true;
        }
      }

      return need_expansion;
    }
    
    static bool ConsiderPairExtrinsic(
        const TkdeParam& param,
        const QNode& q_node,
        const RNode& r_node,
        const TkdeDelta& delta,
        const TkdeMassResult& q_mass_result,
        const TkdeGlobalResult& global_result,
        TkdePostponed* q_postponed) {
      return true;
    }
    
    static bool ConsiderQueryTermination(
        const TkdeParam& param,
        const QNode& q_node,
        const TkdeMassResult& q_mass_result,
        const TkdeGlobalResult& global_result,
        TkdePostponed* q_postponed) {
      bool need_expansion = false;
      
      if (unlikely(q_mass_result.label != LAB_UNKNOWN)) {
        q_postponed->label = q_mass_result.label;
      } else if (unlikely(q_mass_result.density.lo - EPS > param.thresh)) {
        q_postponed->label = LAB_HI;
      } else if (unlikely(q_mass_result.density.hi + EPS < param.thresh)) {
        q_postponed->label = LAB_LO;
      } else {
        need_expansion = true;
      }
      
      return need_expansion;
    }
    
    /**
     * Computes a heuristic for how early a computation should occur -- smaller
     * values are earlier.
     */
    static double Heuristic(
        const TkdeParam& param,
        const QNode& q_node,
        const RNode& r_node,
        const TkdeDelta& delta,
        const TkdeMassResult& q_mass_result) {
      return q_node.bound().MidDistanceSqToBound(r_node.bound());
    }
  };
};
