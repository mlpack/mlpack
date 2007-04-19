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

  struct QInfo {};

  struct RInfo {};

  struct MomentInfo {
    ALLOW_COPY(MomentInfo);
    
   public:
    Vector mass;
    double sumsq;
    index_t count;
    
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
    
    void Postprocess(const TkdeParam& param, const Bound& bound, index_t n) {
    }
  };

  typedef BinarySpaceTree<Bound, Matrix, TkdeStat> RNode;
  typedef BinarySpaceTree<Bound, Matrix, EmptyStatistic<Matrix> > QNode;

  /**
   * Coarse result on a region.
   */
  struct TkdePostponed {
    /** Moments of pruned things. */
    MomentInfo moment_info;

    void Init(const TkdeParam& param) {
      moment_info.Init(param);
    }

    void ApplyPostponed(const TkdeParam& param, const TkdePostponed& other) {
      moment_info.Add(other.moment_info);
    }
  };

  /**
   * Coarse result on a region.
   */
  struct TkdeDelta {
    /** Density update to apply to children's bound. */
    DRange d_density;
    
    void Init(const TkdeParam& param) {
      d_density.Init(0, 0);
    }
    
    void ApplyDelta(const TkdeParam& param, const TkdeDelta& other) {
      d_density += other.d_density;
    }
  };
  
  enum Label {
    LAB_LO = -1,
    LAB_UNKNOWN = 0,
    LAB_HI = 1
  };

  // rho, but a bit of phi and lambda
  struct TkdeResult {
    double density;
    Label label;

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
      if (!postponed.moment_info.is_empty()) {
        density += postponed.moment_info.ComputeKernelSum(param, q_point);
      }
    }
  };

  class TkdeGlobalResult {
    void Init(const TkdeParam& param) {
    }
    
    void Accumulate(const TkdeParam& param,
        const TkdeGlobalResult& other_global_result) {
    }
    
    void ApplyDelta(const TkdeParam& param,
        const TkdeDelta& delta) {
    }
    
    void Postprocess(const TkdeParam& param) {
    }
  };

  struct TkdeMassResult {
    /** Bound on density from leaves. */
    DRange density;
    Label label;

    void Copy(const TkdeMassResult& other) {
      density = other.density;
    }

    void Init(const TkdeParam& param) {
      /* horizontal init */
      density.Init(0, 0);
    }

    void StartReaccumulate(const TkdeParam& param, const QNode& q_node) {
      /* vertical init */
      density.InitEmptySet();
    }

    void Accumulate(const TkdeParam& param, const TkdeResult& result,
        const TkdeMassResult& horizontal_result) {
      // TODO: applying to single result could be made part of Result,
      // but in some cases may require a copy/undo stage
      density |= result.density + horizontal_result.density;
    }

    void Accumulate(const TkdeParam& param,
        const TkdeMassResult& result, index_t n_points) {
      density |= result.density;
    }

    void FinishReaccumulate(const TkdeParam& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }
    
    /** horizontal join operator */
    void ApplyMassResult(const TkdeParam& param,
        const TkdeMassResult& mass_result) {
      density += mass_result.density;
    }

    void ApplyDelta(const TkdeParam& param,
        const TkdeDelta& delta) {
      density += delta.d_density;
    }
    
    bool ApplyPostponed(const TkdeParam& param,
        const TkdePostponed& postponed, const QNode& q_node) {
      bool anything_to_do = !postponed.moment_info.is_empty();
      
      if (unlikely(anything_to_do)) {
        density += moment_info.ComputeKernelSumRange(param, q_node.bound());
      }
      
      return anything_to_do;
    }
  };

  struct TkdeVectorPairVisitor {
    double density;
    
    void Init(const TkdeParam& param) {
    }
    
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
      
      if (unlikely(unapplied_mass_results->density.lo > adjusted_threshold)) {
        q_result->label = LAB_HI;
      }
      
      if (unlikely(unapplied_mass_result->density.hi < adjusted_threshold)) {
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
        const TkdeDelta& delta,
        const TkdeMassResult& q_mass_result,
        const TkdeGlobalResult& global_result,
        TkdePostponed* q_postponed) {
      bool need_expansion = true;
      
      if (q_mass_result.density.lo > param.thresh) {
        q_postponed->label = 1;
        need_expansion = false;
      } else if (q_mass_result.density.hi < param.thresh) {
        q_postponed->label = -1;
        need_expansion = false;
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


/*

depth first

-> each node has....
  -> its mu value (which contains delta)
  -> pushdown deltas

void Recurse(Node *q, List *list_old, ValueResults val_old, PruneResults pr_old) {
  PrunedResults pr_new;
  CoarseResults cr_new;
  ValueResults val_new;
  List *list_new;
  
  for (r in list_old) {
    d_l <- delta(q, r_l)
    if (must_explore(d_l, val_old)) {
      list_new += r_l
      cr_new += d_r
    } else {
      pr_new += d_r
    }
    d_r <- delta(q, r_r)
    if (must_explore(d_r, val_old)) {
      list_new += r_r
      cr_new += d_r
    } else {
      pr_new += d_r
    }
  }
  
  pr_new += pr_old
  val_new = pr_new + cr_new
  
  PrunedResults pr_l, pr_r;
  CoarseResults cr_l, cr_r;
  ValueResults val_l, val_r;
  List *list_l
  List *list_r
  
  for (r in list_new) {
    d_l <- delta(q_l, r)
    if (must_explore(d_l, val_new)) {
      list_l += r
      cr_l += d_l
    } else {
      pr_l += d_l
    }
    
    d_r <- delta(q_r, r)
    if (must_explore(d_r, val_new)) {
      list_r += r
      cr_r += d_r
    } else {
      pr_r += d_r
    }
  }
  
  
  
  val_l = pr_l + cr_l;
  val_r = pr_r + cr_r;
  
  Split(cr_l + )
}
*/
