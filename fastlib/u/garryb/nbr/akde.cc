#include "fastlib/fastlib.h"
#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"
#include "nbr_utils.h"

/*

I spent about a day chasing an elusive bug.  The bug is due to the sloppy
semantics of ConsiderPairTermination and ConsiderPairExtrinsic.

Here's a proposal:

In an extrinsic prune, it is assumed ANY side-effects (such as the delta)
are applied by ConsiderPairExtrinsic to the postponed, rather than by NBR.
On the other hand, a termination prune needs to mark whatever the final
result was.

The bug I found was when I was "horizonally merging" values of Postponed
results, which is really when a postponed result is pushed down to the child.
If there is a postponed "label assignment" (i.e. it was decided the entire
tree had the same label), then it wouldn't apply the kernel sums as it would
seem unnecessary.  Unfortunately, the vertical join operator on mu cares a
lot about the actual densities involved -- it doesn't special-case the
situation when it realizes it has a label assigned and realize not to
care about the density.

The way I solved it ensures that as-correct-as-possible densities reach all
the way down to the points, by forwarding queued-up moment information EVEN
IF a label is already obvious.  It would probably be faster that, in case
a label is propagated, to hard-code the density to conform; i.e. if it
is assigned a label of "LO", then setting the upper and lower bound to
something smaller than threshold.

*/

/**
 * An N-Body-Reduce problem.
 */
class Akde {
 public:
  /** The bounding type. Required by NBR. */
  typedef SpHrectBound<2> Bound;

  typedef SpVectorPoint QPoint;
  typedef SpVectorPoint RPoint;

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
    /** The dimensionality of the data sets. */
    index_t dim;
    /** The epsilon used for error computation. */
    double epsilon;
    /** The epsilon used for error computation, divided evenly. */
    double epsilon_divided;
    /** Normalization factor to be multiplied by later. */
    double post_factor;

    OT_DEF(Param) {
      OT_MY_OBJECT(kernel);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(epsilon);
      OT_MY_OBJECT(epsilon_divided);
      OT_MY_OBJECT(post_factor);
    }

   public:
    void Copy(const Param& other) {
      kernel.Copy(other.kernel);
      dim = other.dim;
      epsilon = other.epsilon;
      epsilon_divided = other.epsilon_divided;
      post_factor = other.post_factor;
    }

    /**
     * Initialize parameters from a data node (Req NBR).
     */
    void Init(datanode *module) {
      kernel.Init(fx_param_double_req(module, "h"));
      epsilon = fx_param_double_req(module, "epsilon");
    }

    void BootstrapMonochromatic(QPoint* point, index_t count) {
      dim = point->vec().length();
      epsilon_divided = 2 * epsilon / count;
      post_factor = 1.0 / kernel.CalcNormConstant(dim) / count / (count - 1);
    }

   public:

    double TwiceEpsilonFor(index_t count) const {
      return epsilon_divided * count;
    }
    
    // Convenience methods

    /**
     * Compute kernel sum for a region of reference points assuming we have
     * the actual query point (not NBR).
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
     * Divides a vector by an integer (not NBR).
     */
    static void ComputeCenter(
        index_t count, const Vector& mass, Vector* center) {
      DEBUG_ASSERT(count != 0);
      center->Copy(mass);
      la::Scale(1.0 / count, center);
    }

    /**
     * Compute kernel sum given only a squared distance (not NBR).
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
    void Accumulate(const Param& param, const QPoint& point) {
      moment_info.Add(1, point.vec(), la::Dot(point.vec(), point.vec()));
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

  /**
   * Query-tree statistic.
   *
   * Note that this statistic is not actually needed and a blank statistic
   * is fine, but QStat must equal RStat in order for us to allow
   * monochromatic execution.
   *
   * This limitation may be removed in a further version of NBR.
   */
  typedef RStat QStat;
 
  /**
   * Query node.
   */
  typedef SpNode<Bound, RStat> RNode;
  /**
   * Reference node.
   */
  typedef SpNode<Bound, QStat> QNode;

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
   public:
    SpRange d_density;

    OT_DEF(QPostponed) {
      OT_MY_OBJECT(d_density);
    }

   public:
    void Init(const Param& param) {
      d_density.Init(0, 0);
    }

    void Reset(const Param& param) {
      d_density.Reset(0, 0);
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      d_density += other.d_density;
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
    }
  };

  // rho
  struct QResult {
   public:
    SpRange density;

    OT_DEF(QResult) {
      OT_MY_OBJECT(density);
    }

   public:
    void Init(const Param& param) {
      density.Init(0, 0);
    }

    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_root) {
      density.hi -= 1; // leave-one-out
      density.lo -= 1; // leave-one-out
      density.hi *= param.post_factor;
      density.lo *= param.post_factor;
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q, index_t q_i) {
      //if (!postponed.moment_info.is_empty()) {
      //  density += postponed.moment_info.ComputeKernelSum(
      //      param, q.vec());
      //}
      density += postponed.d_density;
    }
  };

  struct GlobalResult {
   public:
    SpRange log_likelihood;
    SpRange sum;

   public:
    void Init(const Param& param) {
      log_likelihood.Init(0, 0);
      sum.Init(0, 0);
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
      log_likelihood += other.log_likelihood;
      sum += other.sum;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
      fx_format_result(datanode, "log_likelihood_min", "%g",
          log_likelihood.lo);
      fx_format_result(datanode, "log_likelihood", "%g",
          log_likelihood.mid());
      fx_format_result(datanode, "log_likelihood_max", "%g",
          log_likelihood.hi);
      fx_format_result(datanode, "log_type", "natural_log");
      fx_format_result(datanode, "sum", "%g", sum.mid());
      fx_format_result(datanode, "sum_min", "%g", sum.lo);
      fx_format_result(datanode, "sum_max", "%g", sum.hi);
    }
    /**
     * Used for post-map reductions.
     */
    void ApplyResult(const Param& param,
        const QPoint& q_point, index_t q_i, const QResult& result) {
      log_likelihood.lo += log(result.density.lo);
      log_likelihood.hi += log(result.density.hi);
      sum += result.density;
    }
  };

  struct QSummaryResult {
   public:
    /** Bound on density from leaves. */
    SpRange density;

    OT_DEF(QSummaryResult) {
      OT_MY_OBJECT(density);
    }

   public:
    void Init(const Param& param) {
      /* horizontal init */
      density.Init(0, 0);
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      /* vertical init */
      density.InitEmptySet();
    }

    void Accumulate(const Param& param, const QResult& result) {
      // TODO: applying to single result could be made part of QResult,
      // but in some cases may require a copy/undo stage
      density |= result.density;
    }

    void Accumulate(const Param& param,
        const QSummaryResult& result, index_t n_points) {
      density |= result.density;
    }

    void FinishReaccumulate(const Param& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }

    /** horizontal join operator */
    void ApplySummaryResult(const Param& param,
        const QSummaryResult& summary_result) {
      density += summary_result.density;
    }

    void ApplyDelta(const Param& param,
        const Delta& delta) {
      density += delta.d_density;
    }

    bool ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      //if (unlikely(!postponed.moment_info.is_empty())) {
      //  density += postponed.moment_info.ComputeKernelSumRange
      //            (param, q_node.bound());
      //        return true;
      // }
      // return false;
      density += postponed.d_density;

      return true;
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
        const QPoint& q, index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      // Perform intrinsic prunes here.
      // Don't bother with the error distribution here.
      double distance_sq_lo = r_node.bound().MinDistanceSqToPoint(q.vec());

      if (unlikely(distance_sq_lo > param.kernel.bandwidth_sq())) {
        return false;
      }

      //double distance_sq_hi = r_node.bound().MaxDistanceSqToPoint(q.vec());
      //if (unlikely(distance_sq_hi < param.kernel.bandwidth_sq())) {
      //  q_result->density += r_node.stat().moment_info.ComputeKernelSum(
      //            param, q.vec());
      //        return false;
      //}

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
        const QPoint& q, index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      q_result->density += density;
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
      
      
      if (distance_sq_lo > param.kernel.bandwidth_sq()) {
        need_expansion = false;
      } else {
        double distance_sq_hi =
            q_node.bound().MaxDistanceSqToBound(r_node.bound());

        //if (distance_sq_hi < param.kernel.bandwidth_sq()) {
        //  DEBUG_MSG(1.0, "tkde: Inclusion");
        //  q_postponed->moment_info.Add(r_node.stat().moment_info);
        //  need_expansion = false;
        //} else {
        //  .. set delta lo to 0, hi to squared
        //}

        delta->d_density.lo = r_node.count() *
            param.kernel.EvalUnnormOnSq(distance_sq_hi);
        delta->d_density.hi = r_node.count() *
            param.kernel.EvalUnnormOnSq(distance_sq_lo);
        need_expansion = true;
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
      if (delta.d_density.width() <
          param.TwiceEpsilonFor(r_node.count()) * q_summary_result.density.lo) {
        q_postponed->d_density += delta.d_density;
      }
      return true;
    }

    static bool ConsiderQueryTermination(
        const Param& param,
        const QNode& q_node,
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
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
  nbr_utils::MpiDualTreeMain<Akde, DualTreeDepthFirst<Akde> >(
      fx_root, "akde");
  MPI_Finalize();
#else
  nbr_utils::MonochromaticDualTreeMain<Akde, DualTreeDepthFirst<Akde> >(
      fx_root, "akde");
#endif
  
  fx_done();
}

