#include "fastlib/fastlib.h"
#include "spbounds.h"
#include "gnp.h"

/**
 * An N-Body-Reduce problem.
 */
class Tkde {
 public:
  /** The bounding type. Required by NBR. */
  typedef SpHrectBound<2> Bound;
  /** The type of point in use. Required by NBR. */
  typedef Vector Point;

  /** The type of kernel in use.  NOT required by NBR. */
  typedef EpanKernel Kernel;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by N-Body Reduce.
   */
  struct Param {
   public:
    /** The kernel in use. */
    Kernel kernel;
    /**
     * The threshold in use.
     * This is a range to allow for epsilon checking.
     */
    SpRange thresh;
    /** The dimensionality of the data sets. */
    index_t dim;

    OT_DEF(Param) {
      OT_MY_OBJECT(kernel);
      OT_MY_OBJECT(dim);
    }

   public:
    /**
     * Initialize parameters from a data node (Req NBR).
     */
    void Init(datanode *datanode, const Matrix& q_matrix,
        const Matrix& r_matrix) {
      dim = q_matrix.n_rows();
      kernel.Init(fx_param_double_req(datanode, "h"));
      double t = fx_param_double_req(datanode, "threshold");
      t = t * kernel.CalcNormConstant(dim);
      fx_format_result(datanode, "norm_constant", "%f",
          kernel.CalcNormConstant(dim));
      thresh.lo = t * (1.0 - 1.0e-4);
      thresh.hi = t * (1.0 + 1.0e-4);
      // WALDO: Fix me
    }

   public:
    // Convenience methods for purpose of thresholded KDE

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
      //q*q - 2qr + rsumsq
      //q*q - 2qr + r*r - r*r
      double quadratic_term =
          (distance_squared - la::Dot(r_center, r_center)) * r_count
          + r_sumsq;

      return -quadratic_term * kernel.inv_bandwidth_sq() + r_count;
    }
  };

  /**
   * Per-point extra information, which in case of TKDE is blank.
   *
   * For KDE this might be weights.
   *
   * Not required by N-Body Reduce, although QPointInfo and RPointInfo are.
   */
  struct BlankInfo {
    OT_DEF(BlankInfo) {}
  };

  /** Per-query-point input information.  Required by NBR. */
  typedef BlankInfo QPointInfo;
  /** Per-reference-point input information.  Required by NBR. */
  typedef BlankInfo RPointInfo;

  /**
   * Moment information used by thresholded KDE.
   *
   * NOT required by NBR, but used within other classes.
   */
  struct MomentInfo {
   public:
    Vector mass;
    double sumsq;
    index_t count;

    OT_DEF(MomentInfo) {
      OT_MY_OBJECT(mass);
      OT_MY_OBJECT(sumsq);
      OT_MY_OBJECT(count);
    }

   public:
    void Init(const Param& param) {
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

    double ComputeKernelSum(const Param& param, const Vector& point) const {
      return param.ComputeKernelSum(point, count, mass, sumsq);
    }

    SpRange ComputeKernelSumRange(const Param& param,
        const Bound& query_bound) const {
      SpRange density_bound;
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

  /**
   * Per-reference-node bottom-up statistic.
   *
   * The statistic must be commutative and associative, thus bottom-up
   * computable.
   */
  struct RStat {
   public:
    MomentInfo moment_info;

    OT_DEF(RStat) {
      OT_MY_OBJECT(moment_info);
    }

   public:
    /**
     * Initialize to a default zero value, as if no data is seen (Req NBR).
     *
     * This is the only method in which memory allocation can occur.
     */
    void Init(const Param& param) {
      moment_info.Init(param);
    }

    /**
     * Accumulate data from a single point (Req NBR).
     */
    void Accumulate(const Param& param, const Vector& point,
        const RPointInfo& r_info) {
      moment_info.Add(1, point, la::Dot(point, point));
    }

    /**
     * Accumulate data from one of your children (Req NBR).
     */
    void Accumulate(const Param& param,
        const RStat& stat, const Bound& bound, index_t n) {
      moment_info.Add(stat.moment_info);
    }

    /**
     * Finish accumulating data; for instance, for mean, divide by the
     * number of points.
     */
    void Postprocess(const Param& param, const Bound& bound, index_t n) {
    }
  };

  class BlankStat {
   public:
    void Init(const Param& param) {
    }

    void Accumulate(const Param& param, const Vector& point,
        const QPointInfo& r_info) {
    }

    void Accumulate(const Param& param,
        const BlankStat& stat, const Bound& bound, index_t n) {
    }

    void Postprocess(const Param& param, const Bound& bound, index_t n) {
    }
  };
  
  typedef BlankStat QStat;
  
  /**
   * Query node.
   */
  typedef SpNode<Bound, RStat> RNode;
  /**
   * Reference node.
   */
  typedef SpNode<Bound, QStat> QNode;

  enum Label {
    LAB_LO = 2,
    LAB_UNKNOWN = 0,
    LAB_HI = 1,
    LAB_CONFLICT = 3
  };

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
   public:
    /** Moments of pruned things. */
    MomentInfo moment_info;
    /** We pruned an entire part of the tree with a particular label. */
    int label;

    OT_DEF(QPostponed) {
      OT_MY_OBJECT(moment_info);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      moment_info.Init(param);
      label = LAB_UNKNOWN;
    }

    void Reset(const Param& param) {
      moment_info.Reset();
      label = LAB_UNKNOWN;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      label |= other.label;
      DEBUG_ASSERT_MSG(label != LAB_CONFLICT, "Conflicting labels?");
      moment_info.Add(other.moment_info);
    }
  };

  /**
   * Coarse result on a region.
   */
  struct Delta {
   public:
    /** Density update to apply to children's bound. */
    SpRange d_density;

    OT_DEF(Delta) {
      OT_MY_OBJECT(d_density);
    }

   public:
    void Init(const Param& param) {
      d_density.Init(0, 0);
    }
  };

  // rho
  struct QResult {
   public:
    double density;
    int label;

    OT_DEF(QResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param,
        const Vector& q_point, const QPointInfo& q_info,
        const RNode& r_root) {
      density = 0;
      label = LAB_UNKNOWN;
    }

    void Postprocess(const Param& param,
        const Vector& q_point, const QPointInfo& q_info,
        const RNode& r_root) {
      if (density > param.thresh.hi) {
        label |= LAB_HI;
      } else if (density < param.thresh.lo) {
        label |= LAB_LO;
      }
      DEBUG_ASSERT(label != LAB_CONFLICT);
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const Vector& q_point) {
      label |= postponed.label; /* bitwise OR */
      DEBUG_ASSERT_MSG(label >= 0 && label < 3, "%d", label);

      if (!postponed.moment_info.is_empty()) {
        density += postponed.moment_info.ComputeKernelSum(param, q_point);
      }
    }
  };

  class GlobalResult {
   public:
    OT_DEF(GlobalResult) {}

   public:
    void Init(const Param& param) {}
    void Accumulate(const Param& param,
        const GlobalResult& other_global_result) {}
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}
  };

  struct QMassResult {
   public:
    /** Bound on density from leaves. */
    SpRange density;
    int label;

    OT_DEF(QMassResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      /* horizontal init */
      density.Init(0, 0);
      label = LAB_UNKNOWN;
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      /* vertical init */
      density.InitEmptySet();
      label = LAB_CONFLICT;
    }

    void Accumulate(const Param& param, const QResult& result) {
      // TODO: applying to single result could be made part of QResult,
      // but in some cases may require a copy/undo stage
      density |= result.density;
      label &= result.label;
      DEBUG_ASSERT(result.label != LAB_CONFLICT);
    }

    void Accumulate(const Param& param,
        const QMassResult& result, index_t n_points) {
      density |= result.density;
      label &= result.label;
      DEBUG_ASSERT(result.label != LAB_CONFLICT);
    }

    void FinishReaccumulate(const Param& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }

    /** horizontal join operator */
    void ApplyMassResult(const Param& param,
        const QMassResult& mass_result) {
      density += mass_result.density;
      DEBUG_ASSERT_MSG((label | mass_result.label) != LAB_CONFLICT,
          "%d and %d", label, mass_result.label);
      label |= mass_result.label;
    }

    void ApplyDelta(const Param& param,
        const Delta& delta) {
      density += delta.d_density;
    }

    bool ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      bool change_made;

      if (unlikely(postponed.label)) {
        DEBUG_ASSERT((label | postponed.label) != LAB_CONFLICT);
        label = postponed.label;
        change_made = true;
      } else  if (unlikely(!postponed.moment_info.is_empty())) {
        density += postponed.moment_info.ComputeKernelSumRange
            (param, q_node.bound());
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
  struct PairVisitor {
   public:
    double density;

   public:
    void Init(const Param& param) {}

    // notes
    // - this function must assume that global_result is incomplete (which is
    // reasonable in allnn)
    bool StartVisitingQueryPoint(const Param& param,
        const Vector& q_point,
        const QPointInfo& q_info,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
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

    void VisitPair(const Param& param,
        const Vector& q_point, const QPointInfo& q_info, index_t q_index,
        const Vector& r_point, const RPointInfo& r_info, index_t r_index) {
      double distance = la::DistanceSqEuclidean(q_point, r_point);
      density += param.kernel.EvalUnnormOnSq(distance);
    }

    void FinishVisitingQueryPoint(const Param& param,
        const Vector& q_point,
        const QPointInfo& q_info,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      q_result->density += density;

      if (unlikely(unapplied_mass_results.density.lo + q_result->density
          > param.thresh.hi)) {
        q_result->label = LAB_HI;
      } else if (unlikely(unapplied_mass_results.density.hi + q_result->density
          < param.thresh.lo)) {
        q_result->label = LAB_LO;
      }
    }
  };

  class Algorithm {
   public:
    /**
     * Calculates a delta....
     *
     * - If this returns true, delta is calculated, and global_result is
     * updated.  q_postponed is not touched.
     * - If this returns false, delta is not touched.
     */
    static bool ConsiderPairIntrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        Delta* delta,
        GlobalResult* global_result,
        QPostponed* q_postponed) {
      double distance_sq_lo =
          q_node.bound().MinDistanceSqToBound(r_node.bound());
      bool need_expansion;
      
      //printf("%f %f %f\n",
      //    q_node.bound().MinDistanceSqToBound(r_node.bound()),
      //    q_node.bound().MidDistanceSqToBound(r_node.bound()),
      //    q_node.bound().MaxDistanceSqToBound(r_node.bound())
      //    );

      if (distance_sq_lo > param.kernel.bandwidth_sq()) {
        need_expansion = false;
      } else {
        double distance_sq_hi =
            q_node.bound().MaxDistanceSqToBound(r_node.bound());

        if (distance_sq_hi < param.kernel.bandwidth_sq()) {
          q_postponed->moment_info.Add(r_node.stat().moment_info);
          need_expansion = false;
        } else {
#ifdef BIGPRUNE
#endif
          //delta->d_density = r_node.stat().moment_info.ComputeKernelSumRange(
          //    param, q_node.bound());
          // we computed the lower bound of the quadratic.  if it is positive
          // it means we have a better-than-nothing bound; if it is not, then
          // we can resort to saying the min contribution is zero.
          //    - problem: the upper bound is no good.
          //max(delta->d_density.lo, 0.0);
          delta->d_density.lo = 0;
          delta->d_density.hi = r_node.count() *
              param.kernel.EvalUnnormOnSq(distance_sq_lo);
          need_expansion = true;
#ifdef BIGPRUNE
#endif
        }
      }

      return need_expansion;
    }

    static bool ConsiderPairExtrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta,
        const QMassResult& q_mass_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }

    static bool ConsiderQueryTermination(
        const Param& param,
        const QNode& q_node,
        const QMassResult& q_mass_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      bool need_expansion = false;

      if (unlikely(q_mass_result.label != LAB_UNKNOWN)) {
        q_postponed->label = q_mass_result.label;
      } else if (unlikely(q_mass_result.density.lo > param.thresh.hi)) {
        q_postponed->label = LAB_HI;
      } else if (unlikely(q_mass_result.density.hi < param.thresh.lo)) {
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
        const Param& param,
        const QNode& q_node,
        const RNode& r_node) {
      return q_node.bound().MidDistanceSqToBound(r_node.bound());
    }
  };
  
  typedef DualTreeGNP<
      Param, Algorithm,
      Point, Bound,
      QPointInfo, QStat,
      RPointInfo, RStat,
      PairVisitor, Delta,
      QResult, QMassResult, QPostponed,
      GlobalResult>
    GNP;
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  
  DualTreeDepthFirst<Tkde::GNP> dfs;
  dfs.Init(fx_submodule(fx_root, "dfs", "dfs"));
  dfs.Begin();
  
  fx_done();
}

