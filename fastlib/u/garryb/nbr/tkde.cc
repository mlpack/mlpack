#include "fastlib/fastlib.h"
#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"
#include "thor_utils.h"

/**
 * Approximate kernel density estimation.
 *
 * Uses Epanechnikov kernels and the inclusion rule.
 */
class Tkde {
 public:
  /** The bounding type. Required by THOR. */
  typedef HrectBound<2> Bound;

  typedef ThorVectorPoint QPoint;
  typedef ThorVectorPoint RPoint;

  /** The type of kernel in use.  NOT required by THOR. */
  typedef EpanKernel Kernel;

  typedef BlankGlobalResult GlobalResult;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by N-Body Reduce.
   */
  struct Param {
   public:
    /**
     * The threshold in use, with upper and lower bounds to prevent
     * roundoff error.
     *
     * This is also normalized for dimensionality and the number of reference
     * points.
     */
    DRange thresh;
    /** The kernel in use. */
    Kernel kernel;
    /** The dimensionality of the data sets. */
    index_t dim;
    /** The original threshold, before normalization. */
    double nominal_threshold;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(thresh);
      OT_MY_OBJECT(kernel);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(nominal_threshold);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      kernel.Init(fx_param_double_req(module, "h"));
      nominal_threshold = fx_param_double_req(module, "threshold");
    }

    void BootstrapMonochromatic(QPoint* point, index_t count) {
      dim = point->vec().length();
      double normalized_threshold =
          nominal_threshold * kernel.CalcNormConstant(dim) * count;
      thresh.lo = normalized_threshold * (1.0 - 1.0e5);
      thresh.hi = normalized_threshold * (1.0 + 1.0e5);
    }

   public:
    // Convenience methods for purpose of thresholded KDE

    /**
     * Compute kernel sum for a region of reference points assuming we have
     * the actual query point (not THOR).
     */
    double ComputeKernelSum(
        const Vector& q,
        index_t r_count, const Vector& r_mass, double r_sumsq) const {
      double quadratic_term =
          + r_count * la::Dot(q, q)
          - 2.0 * la::Dot(q, r_mass)
          + r_sumsq;
      return r_count - quadratic_term * kernel.inv_bandwidth_sq();
    }

    /**
     * Divides a vector by an integer (not THOR).
     */
    static void ComputeCenter(
        index_t count, const Vector& mass, Vector* center) {
      DEBUG_ASSERT(count != 0);
      center->Copy(mass);
      la::Scale(1.0 / count, center);
    }

    /**
     * Compute kernel sum given only a squared distance (not THOR).
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
   * Moment information used by thresholded KDE.
   *
   * NOT required by THOR, but used within other classes.
   */
  struct MomentInfo {
   public:
    Vector mass;
    double sumsq;
    index_t count;

    OT_DEF_BASIC(MomentInfo) {
      OT_MY_OBJECT(mass);
      OT_MY_OBJECT(sumsq);
      OT_MY_OBJECT(count);
    }

   public:
    void Init(const Param& param) {
      DEBUG_ASSERT(param.dim > 0);
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

    DRange ComputeKernelSumRange(const Param& param,
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

  /**
   * Per-reference-node bottom-up statistic.
   *
   * The statistic must be commutative and associative, thus bottom-up
   * computable.
   */
  struct RStat {
   public:
    MomentInfo moment_info;

    OT_DEF_BASIC(RStat) {
      OT_MY_OBJECT(moment_info);
    }

   public:
    /**
     * Initialize to a default zero value, as if no data is seen (Req THOR).
     *
     * This is the only method in which memory allocation can occur.
     */
    void Init(const Param& param) {
      moment_info.Init(param);
    }

    /**
     * Accumulate data from a single point (Req THOR).
     */
    void Accumulate(const Param& param, const QPoint& point) {
      moment_info.Add(1, point.vec(), la::Dot(point.vec(), point.vec()));
    }

    /**
     * Accumulate data from one of your children (Req THOR).
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

  /**
   * Query-tree statistic.
   *
   * Note that this statistic is not actually needed and a blank statistic
   * is fine, but QStat must equal RStat in order for us to allow
   * monochromatic execution.
   *
   * This limitation may be removed in a further version of THOR.
   */
  typedef RStat QStat;
 
  /**
   * Query node.
   */
  typedef ThorNode<Bound, RStat> RNode;
  /**
   * Reference node.
   */
  typedef ThorNode<Bound, QStat> QNode;

  enum Label {
    LAB_NEITHER = 0,
    LAB_LO = 2,
    LAB_HI = 1,
    LAB_EITHER = 3
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

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(moment_info);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      moment_info.Init(param);
      label = LAB_EITHER;
    }

    void Reset(const Param& param) {
      moment_info.Reset();
      label = LAB_EITHER;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      label &= other.label;
      DEBUG_ASSERT_MSG(label != LAB_NEITHER, "Conflicting labels?");
      moment_info.Add(other.moment_info);
    }
  };

  /**
   * Coarse result on a region.
   */
  struct Delta {
   public:
    /** Density update to apply to children's bound. */
    DRange d_density;

    OT_DEF_BASIC(Delta) {
      OT_MY_OBJECT(d_density);
    }

   public:
    void Init(const Param& param) {
    }
  };

  // rho
  struct QResult {
   public:
    double density;
    int label;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      density = 0;
      label = LAB_EITHER;
    }

    void Postprocess(const Param& param,
        const QPoint& q,
        const RNode& r_root) {
      if (density > param.thresh.hi) {
        label &= LAB_HI;
      } else if (density < param.thresh.lo) {
        label &= LAB_LO;
      }
      DEBUG_ASSERT(label != LAB_NEITHER);
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q) {
      label &= postponed.label; /* bitwise OR */
      DEBUG_ASSERT(label != LAB_NEITHER);

      if (!postponed.moment_info.is_empty()) {
        density += postponed.moment_info.ComputeKernelSum(
            param, q.vec());
      }
    }
  };

  struct QSummaryResult {
   public:
    /** Bound on density from leaves. */
    DRange density;
    int label;

    OT_DEF_BASIC(QSummaryResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(label);
    }

   public:
    void Init(const Param& param) {
      /* horizontal init */
      density.Init(0, 0);
      label = LAB_EITHER;
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      /* vertical init */
      density.InitEmptySet();
      label = LAB_NEITHER;
    }

    void Accumulate(const Param& param, const QResult& result) {
      // TODO: applying to single result could be made part of QResult,
      // but in some cases may require a copy/undo stage
      density |= result.density;
      label |= result.label;
      DEBUG_ASSERT(result.label != LAB_NEITHER);
    }

    void Accumulate(const Param& param,
        const QSummaryResult& result, index_t n_points) {
      density |= result.density;
      label |= result.label;
      DEBUG_ASSERT(result.label != LAB_NEITHER);
    }

    void FinishReaccumulate(const Param& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }

    /** horizontal join operator */
    void ApplySummaryResult(const Param& param,
        const QSummaryResult& summary_result) {
      density += summary_result.density;
      label &= summary_result.label;
      DEBUG_ASSERT(label != LAB_NEITHER);
    }

    void ApplyDelta(const Param& param,
        const Delta& delta) {
      density += delta.d_density;
    }

    bool ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      bool change_made;

      if (unlikely(postponed.label != LAB_EITHER)) {
        label &= postponed.label;
        DEBUG_ASSERT(label != LAB_NEITHER);
        change_made = true;
      }
      if (unlikely(!postponed.moment_info.is_empty())) {
        density += postponed.moment_info.ComputeKernelSumRange
            (param, q_node.bound());
        change_made = true;
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
        const QPoint& q,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      if (unlikely(q_result->label != LAB_EITHER)) {
        return false;
      }

      double distance_sq_lo = r_node.bound().MinDistanceSqToPoint(q.vec());

      if (unlikely(distance_sq_lo > param.kernel.bandwidth_sq())) {
        return false;
      }

      double distance_sq_hi = r_node.bound().MaxDistanceSqToPoint(q.vec());

      if (unlikely(distance_sq_hi < param.kernel.bandwidth_sq())) {
        q_result->density += r_node.stat().moment_info.ComputeKernelSum(
            param, q.vec());
        return false;
      }

      density = 0;

      return true;
    }

    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
      double distance = la::DistanceSqEuclidean(q.vec(), r.vec());
      density += param.kernel.EvalUnnormOnSq(distance);
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      q_result->density += density;
      
      DRange total_density =
          unapplied_summary_results.density + q_result->density;

      if (unlikely(total_density > param.thresh)) {
        q_result->label = LAB_HI;
      } else if (unlikely(total_density < param.thresh)) {
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
      
      DEBUG_MSG(1.0, "tkde: ConsiderPairIntrinsic");
      
      if (distance_sq_lo > param.kernel.bandwidth_sq()) {
        DEBUG_MSG(1.0, "tkde: Exclusion");
        need_expansion = false;
      } else {
        double distance_sq_hi =
            q_node.bound().MaxDistanceSqToBound(r_node.bound());

        if (distance_sq_hi < param.kernel.bandwidth_sq()) {
          DEBUG_MSG(1.0, "tkde: Inclusion");
          q_postponed->moment_info.Add(r_node.stat().moment_info);
          need_expansion = false;
        } else {
          DEBUG_MSG(1.0, "tkde: Overlap - need explore");
          //delta->d_density = r_node.stat().moment_info.ComputeKernelSumRange(
          //    param, q_node.bound());
          //max(delta->d_density.lo, 0.0);
          // this method seemed like a good idea, but the upper bound it
          // computes is unfortunately bogus
          delta->d_density.lo = 0;
          delta->d_density.hi = r_node.count() *
              param.kernel.EvalUnnormOnSq(distance_sq_lo);
          need_expansion = true;
        }
      }

      return need_expansion;
    }

    static bool ConsiderPairExtrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta,
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }

    static bool ConsiderQueryTermination(
        const Param& param,
        const QNode& q_node,
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      bool need_expansion = false;

      DEBUG_ASSERT(q_summary_result.density.lo < q_summary_result.density.hi);

      if (unlikely(q_summary_result.label != LAB_EITHER)) {
        DEBUG_ASSERT((q_summary_result.label & q_postponed->label) != LAB_NEITHER);
        q_postponed->label = q_summary_result.label;
      } else if (unlikely(q_summary_result.density > param.thresh)) {
        q_postponed->label = LAB_HI;
      } else if (unlikely(q_summary_result.density < param.thresh)) {
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
        const RNode& r_node,
        const Delta& delta) {
      return q_node.bound().MidDistanceSqToBound(r_node.bound());
    }
  };
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

#ifdef USE_MPI  
  MPI_Init(&argc, &argv);
  thor_utils::MpiDualTreeMain<Tkde, DualTreeDepthFirst<Tkde> >(
      fx_root, "tkde");
  MPI_Finalize();
#else
  thor_utils::MonochromaticDualTreeMain<Tkde, DualTreeDepthFirst<Tkde> >(
      fx_root, "tkde");
#endif
  
  fx_done();
}

