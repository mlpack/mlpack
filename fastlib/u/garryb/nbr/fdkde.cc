#include "fastlib/fastlib_int.h"
#include "thor/thor.h"

/**
 * Approximate kernel density estimation.
 *
 * Uses Epanechnikov kernels and the inclusion rule.
 */
template<typename TKernel>
class FdKde {
 public:
  /** The bounding type. Required by THOR. */
  typedef DHrectBound<2> Bound;

  typedef ThorVectorPoint QPoint;
  typedef ThorVectorPoint RPoint;

  /** The type of kernel in use.  NOT required by THOR. */
  typedef TKernel Kernel;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by N-Body Reduce.
   */
  struct Param {
   public:
    /** The kernel in use. */
    Kernel kernel;
    /** The amount of relative error prooportional to local lower bound. */
    double rel_error_local;
    /** The amount of relative error distributed uniformly. */
    double rel_error_global;

    /** The dimensionality of the data sets. */
    index_t dim;
    /** Number of points. */
    index_t count;
    /** The multiplicative norm constant for the kernel. */
    double mul_constant;
    /** The amount of relative error allowed. */
    double rel_error;
    /** Amount of error that is local error */
    double p_local;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(kernel);
      OT_MY_OBJECT(rel_error_local);
      OT_MY_OBJECT(rel_error_global);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(count);
      OT_MY_OBJECT(mul_constant);
      OT_MY_OBJECT(rel_error);
      OT_MY_OBJECT(p_local);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      kernel.Init(fx_param_double_req(module, "h"));
      p_local = fx_param_double(module, "p_local", 0.5);
      rel_error = fx_param_double(module, "rel_error", 0.1);
    }

    void SetDimensions(index_t vector_dimension, index_t n_points) {
      dim = vector_dimension;
      count = n_points;

      mul_constant = 1.0 / (kernel.CalcNormConstant(dim) * (count - 1));
      rel_error_local = rel_error * p_local;
      rel_error_global = rel_error * (1.0 - p_local) / count;
    }

   public:
  };

//  /**
//   * Moment information used by thresholded KDE.
//   *
//   * NOT required by THOR, but used within other classes.
//   */
//  struct MomentInfo {
//   public:
//    Vector mass;
//    double sumsq;
//    index_t count;
//
//    OT_DEF_BASIC(MomentInfo) {
//      OT_MY_OBJECT(mass);
//      OT_MY_OBJECT(sumsq);
//      OT_MY_OBJECT(count);
//    }
//
//   public:
//    void Init(const Param& param) {
//      DEBUG_ASSERT(param.dim > 0);
//      mass.Init(param.dim);
//      Reset();
//    }
//
//    void Reset() {
//      mass.SetZero();
//      sumsq = 0;
//      count = 0;
//    }
//
//    void Add(index_t count_in, const Vector& mass_in, double sumsq_in) {
//      if (unlikely(count_in != 0)) {
//        la::AddTo(mass_in, &mass);
//        sumsq += sumsq_in;
//        count += count_in;
//      }
//    }
//
//    void Add(const MomentInfo& other) {
//      Add(other.count, other.mass, other.sumsq);
//    }
//
//    /**
//     * Compute kernel sum for a region of reference points assuming we have
//     * the actual query point.
//     */
//    double ComputeKernelSum(const Param& param, const Vector& q) const {
//      double quadratic_term =
//          + count * la::Dot(q, q)
//          - 2.0 * la::Dot(q, mass)
//          + sumsq;
//      return count - quadratic_term * param.kernel.inv_bandwidth_sq();
//    }
//
//    double ComputeKernelSum(const Param& param, double distance_squared,
//        double center_dot_center) const {
//      //q*q - 2qr + rsumsq
//      //q*q - 2qr + r*r - r*r
//      double quadratic_term =
//          (distance_squared - center_dot_center) * count
//          + sumsq;
//
//      return -quadratic_term * param.kernel.inv_bandwidth_sq() + count;
//    }
//
//    DRange ComputeKernelSumRange(const Param& param,
//        const Bound& query_bound) const {
//      DRange density_bound;
//      Vector center;
//      double inv_count = 1.0 / count;
//      double center_dot_center = la::Dot(mass, mass) * inv_count * inv_count;
//      
//      DEBUG_ASSERT(count != 0);
//
//      center.Copy(mass);
//      for (index_t i = center.length(); i--;) {
//        center[i] *= inv_count;
//      }
//
//      density_bound.lo = ComputeKernelSum(param,
//          query_bound.MaxDistanceSq(center), center_dot_center);
//      density_bound.hi = ComputeKernelSum(param,
//          query_bound.MinDistanceSq(center), center_dot_center);
//
//      return density_bound;
//    }
//
//    bool is_empty() const {
//      return likely(count == 0);
//    }
//  };

  /**
   * Per-reference-node bottom-up statistic.
   *
   * The statistic must be commutative and associative, thus bottom-up
   * computable.
   */
  struct RStat {
   public:
    //MomentInfo moment_info;

    OT_DEF_BASIC(RStat) {
      //OT_MY_OBJECT(moment_info);
    }

   public:
    /**
     * Initialize to a default zero value, as if no data is seen (Req THOR).
     *
     * This is the only method in which memory allocation can occur.
     */
    void Init(const Param& param) {
      //moment_info.Init(param);
    }

    /**
     * Accumulate data from a single point (Req THOR).
     */
    void Accumulate(const Param& param, const QPoint& point) {
      //moment_info.Add(1, point.vec(), la::Dot(point.vec(), point.vec()));
    }

    /**
     * Accumulate data from one of your children (Req THOR).
     */
    void Accumulate(const Param& param,
        const RStat& stat, const Bound& bound, index_t n) {
      //moment_info.Add(stat.moment_info);
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

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
   public:
    /** The density contribution postponed. */
    DRange d_density;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(d_density);
    }

   public:
    void Init(const Param& param) {
      d_density.Init(0, 0);
    }

    void Reset(const Param& param) {
      d_density.Init(0, 0);
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
    /** Min and max density update possible. */
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
    DRange density;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density);
    }

   public:
    void Init(const Param& param) {
      density.Init(0, 0);
    }

    void Postprocess(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_root) {
      density -= 1; // LOO
      density.lo *= param.mul_constant;
      density.hi *= param.mul_constant;
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q, index_t q_index) {
      density += postponed.d_density;
    }
  };

  struct QSummaryResult {
   public:
    /** Bound on density from leaves. */
    DRange density;

    OT_DEF_BASIC(QSummaryResult) {
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

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      density += postponed.d_density;
    }
  };

  /**
   * A simple postprocess-step global result.
   */
  struct GlobalResult {
   public:
    DRange sum_density;
    double foo;
    
    OT_DEF(GlobalResult) {
      OT_MY_OBJECT(sum_density);
      OT_MY_OBJECT(foo);
    }

   public:
    void Init(const Param& param) {
      sum_density.Init(0, 0);
      foo = 0;
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
      sum_density += other.sum_density;
      foo += other.foo;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
      fx_format_result(datanode, "avg_density_lo", "%g", sum_density.lo / param.count);
      fx_format_result(datanode, "avg_density_hi", "%g", sum_density.hi / param.count);
      fx_format_result(datanode, "avg_density", "%g", sum_density.mid() / param.count);
      fx_format_result(datanode, "foo", "%g", foo / param.count);
    }
    void ApplyResult(const Param& param,
        const QPoint& q_point, index_t q_i,
        const QResult& result) {
      sum_density += result.density;
      foo += log(result.density.mid());
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
      double distance_sq_lo = r_node.bound().MinDistanceSq(q.vec());
      double distance_sq_hi = r_node.bound().MaxDistanceSq(q.vec());
      DRange d_density;

      d_density.lo = r_node.count() *
          param.kernel.EvalUnnormOnSq(distance_sq_hi);
      d_density.hi = r_node.count() *
          param.kernel.EvalUnnormOnSq(distance_sq_lo);

      double summary_density_lo = unapplied_summary_results.density.lo
          + d_density.lo + q_result->density.lo;
      double allocated_error =
          param.rel_error_local * d_density.lo
          + param.rel_error_global * summary_density_lo * r_node.count();

      if (d_density.width() < allocated_error * 2) {
        q_result->density += d_density;
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
      //DRange distance_sq =
      //    q_node.bound().RangeDistanceSq(r_node.bound());
      //double distance_sq_mid = q_node.bound().MaxToMidSq(r_node.bound());
      double distance_sq_lo = q_node.bound().MinDistanceSq(r_node.bound());
      double distance_sq_hi = q_node.bound().MaxDistanceSq(r_node.bound());

      delta->d_density.lo = r_node.count() *
          param.kernel.EvalUnnormOnSq(distance_sq_hi);
      delta->d_density.hi = r_node.count() *
          param.kernel.EvalUnnormOnSq(distance_sq_lo);

      return likely(delta->d_density.hi != 0);
    }

    static bool ConsiderPairExtrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta,
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      double allocated_error =
          param.rel_error_local * delta.d_density.lo
          + param.rel_error_global * q_summary_result.density.lo * r_node.count();

      if (delta.d_density.width() < allocated_error * 2) {
        q_postponed->d_density += delta.d_density;
        return false;
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
      //double d = r_node.bound().MinToMidSq(q_node.bound());
      //return delta.d_density.lo / delta.d_density.width() + d * 1.0e-32;
      double d = r_node.bound().MidDistanceSq(q_node.bound());
      return d;
      //return fabs(d - param.kernel.bandwidth_sq());
    }
  };
};

void FdKdeMain(datanode *module) {
  String kernel = fx_param_str(module, "fdkde/kernel", "gauss");
  
  if (kernel.StartsWith("gauss")) {
    thor::MonochromaticDualTreeMain<
        FdKde<GaussianKernel>, DualTreeDepthFirst<FdKde<GaussianKernel> > >(
        module, "fdkde");
  } else if (kernel.StartsWith("epan")) {
    thor::MonochromaticDualTreeMain<
        FdKde<EpanKernel>, DualTreeDepthFirst<FdKde<EpanKernel> > >(
        module, "fdkde");
  }
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  FdKdeMain(fx_root);
  
  fx_done();
}

