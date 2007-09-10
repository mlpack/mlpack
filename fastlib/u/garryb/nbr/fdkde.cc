#include "fastlib/fastlib_int.h"
#include "thor/thor.h"

#define SOLVER_TYPE DualTreeRecursiveBreadth
//#define SOLVER_TYPE DualTreeDepthFirst

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

  typedef ThorVectorPoint Point;
  typedef Point QPoint;
  typedef Point RPoint;

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
    // /** The amount of relative error prooportional to local lower bound. */
    // double rel_error_local;

    /** The dimensionality of the data sets. */
    index_t dim;
    /** Number of points. */
    index_t count;
    /** The multiplicative norm constant for the kernel. */
    double mul_constant;
    /** The amount of relative error allowed. */
    double rel_error;
    // /** Amount of error that is local error */
    // double p_local;
    // /** Amount of error that is global error */
    // double p_global;
    /** The band width, h. */
    double bandwidth;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(kernel);
      //OT_MY_OBJECT(rel_error_local);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(count);
      OT_MY_OBJECT(mul_constant);
      OT_MY_OBJECT(rel_error);
      //OT_MY_OBJECT(p_local);
      //OT_MY_OBJECT(p_global);
      OT_MY_OBJECT(bandwidth);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      bandwidth = fx_param_double_req(module, "h");
      //p_local = fx_param_double(module, "p_local", 0);
      //p_global = 1 - p_local;
      rel_error = fx_param_double(module, "rel_error", 0.1);
      //rel_error_local = rel_error * p_local;
    }

    /** this is called after things are set. */
    void SetDimensions() {
      kernel.Init(bandwidth, dim);
      mul_constant = 1.0 / (kernel.CalcNormConstant(dim) * (count - 1));
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

  typedef ThorNode<Bound, RStat> Node;
  typedef Node QNode;
  typedef Node RNode;

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
   public:
    /** The density contribution postponed. */
    DRange d_density;
    index_t n_pruned;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(d_density);
      OT_MY_OBJECT(n_pruned);
    }

   public:
    void Init(const Param& param) {
      d_density.Init(0, 0);
      n_pruned = 0;
    }

    void Reset(const Param& param) {
      d_density.Init(0, 0);
      n_pruned = 0;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      d_density += other.d_density;
      n_pruned += other.n_pruned;
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
    index_t n_pruned;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(n_pruned);
    }

   public:
    void Init(const Param& param) {
      density.Init(0, 0);
      n_pruned = 0;
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
      n_pruned += postponed.n_pruned;
    }
  };

  struct QSummaryResult {
   public:
    /** Bound on density from leaves. */
    DRange density;
    double used_width;
    index_t n_pruned;

    OT_DEF_BASIC(QSummaryResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(used_width);
      OT_MY_OBJECT(n_pruned);
    }

   public:
    void Init(const Param& param) {
      /* horizontal init */
      density.Init(0, 0);
      used_width = 0;
      n_pruned = 0;
    }

    /** horizontal join operator */
    void ApplySummaryResult(const Param& param,
        const QSummaryResult& summary_result) {
      density += summary_result.density;
      used_width += summary_result.used_width;
      n_pruned += summary_result.n_pruned;
    }

    void ApplyDelta(const Param& param,
        const Delta& delta) {
      density += delta.d_density;
      // deltas don't affect used error
    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {
      density += postponed.d_density;
      used_width += postponed.d_density.width();
      n_pruned += postponed.n_pruned;
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      /* vertical init */
      density.InitEmptySet();
      used_width = 0;
      n_pruned = param.count;
    }

    void Accumulate(const Param& param, const QResult& result) {
      // TODO: applying to single result could be made part of QResult,
      // but in some cases may require a copy/undo stage
      density |= result.density;
      used_width = max(used_width, result.density.width());
      n_pruned = min(n_pruned, result.n_pruned);
    }

    void Accumulate(const Param& param,
        const QSummaryResult& result, index_t n_points) {
      density |= result.density;
      used_width = max(used_width, result.used_width);
      n_pruned = min(n_pruned, result.n_pruned);
    }

    void FinishReaccumulate(const Param& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
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
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
      fx_format_result(datanode, "avg_density_lo", "%g", sum_density.lo / param.count);
      fx_format_result(datanode, "avg_density_hi", "%g", sum_density.hi / param.count);
      fx_format_result(datanode, "avg_density", "%g", sum_density.mid() / param.count);
      fx_format_result(datanode, "avg_rel_error", "%g",
          sum_density.width() / sum_density.lo / 2);
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
      DRange distance_sq_range = DRange(
          r_node.bound().MinDistanceSq(q.vec()),
          r_node.bound().MaxDistanceSq(q.vec()));
      DRange d_density = param.kernel.RangeUnnormOnSq(distance_sq_range);

      d_density *= r_node.count();

      double density_lo = d_density.lo + q_result->density.lo
          + unapplied_summary_results.density.lo;

      double allocated_width =
          (param.rel_error * density_lo * 2 - q_result->density.width())
          * r_node.count() / (param.count - q_result->n_pruned);
      /*allocated_error *= param.p_global;
      allocated_error += param.rel_error_local * d_density.lo;*/

      q_result->n_pruned += r_node.count();

      if (d_density.width() < allocated_width) {
        q_result->density += d_density;
        return false;
      }

      density = 0;

      return true;
    }

    /**
     * This is the lame form of the function used by breadth-first.
     *
     * Since breadth-first tries to avoid getting to leaves anyways, it
     * doesn't want to bother with giving you summary results, so it doesn't.
     */
    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node,
        QResult* q_result,
        GlobalResult* global_result) {
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
        const QPoint& q, index_t q_index, const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result, GlobalResult* global_result) {
      q_result->density += density;
    }

    /**
     * Once again, the lame form for breadth-first.
     */
    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index, const RNode& r_node,
        QResult* q_result, GlobalResult* global_result) {
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
      DRange distance_sq_range = q_node.bound().RangeDistanceSq(r_node.bound());

      delta->d_density = param.kernel.RangeUnnormOnSq(distance_sq_range);
      delta->d_density *= r_node.count();

      DEBUG_ASSERT_MSG(delta->d_density.lo <= delta->d_density.hi * (1 + 1.0e-7),
          "delta density lo %f > hi %f",
          delta->d_density.lo, delta->d_density.hi);

      if (likely(delta->d_density.hi != 0)) {
        return true;
      } else {
        q_postponed->n_pruned += r_node.count();
        return false;
      }
    }

    static bool ConsiderPairExtrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta,
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      double allocated_width =
          (param.rel_error * q_summary_result.density.lo * 2
              - q_summary_result.used_width)
          * r_node.count() / (param.count - q_summary_result.n_pruned);
      /*allocated_width *= param.p_global;
      allocated_width += param.rel_error_local * delta.d_density.lo * 2;*/

      if (delta.d_density.width() < allocated_width) {
        q_postponed->d_density += delta.d_density;
        q_postponed->n_pruned += r_node.count();
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

  static void DoKde(datanode *module) {
    rpc::Init();
    
    if (!rpc::is_root()) {
      // turn off fastexec output
      fx_silence();
    }

    const int TREE_CHANNEL = 300;
    const int RESULT_CHANNEL = 350;
    Param *param;
    ThorTree<Param, Point, Node> tree;
    DistributedCache results;
    double results_megs = fx_param_double(module, "results/megs", 1000);

    param = new Param();
    param->Init(fx_submodule(module, "kde", "kde"));

    fx_timer_start(module, "read");
    DistributedCache *points_cache = new DistributedCache();
    param->count = thor::ReadPoints<Point>(
        Empty(), TREE_CHANNEL + 0, TREE_CHANNEL + 1,
        fx_submodule(module, "data", "data"), points_cache);
    fx_timer_stop(module, "read");

    Point example_point;
    CacheArray<Point>::GetDefaultElement(points_cache,
        &example_point);
    param->dim = example_point.vec().length();
    
    param->SetDimensions();

    fx_timer_start(module, "tree");
    thor::CreateKdTree<Point, Node>(
        *param, TREE_CHANNEL + 2, TREE_CHANNEL + 3,
        fx_submodule(module, "tree", "tree"), param->count,
        points_cache, &tree);
    fx_timer_stop(module, "tree");

    QResult example_result;
    example_result.Init(*param);
    tree.CreateResultCache(RESULT_CHANNEL, example_result,
        results_megs, &results);

    GlobalResult global_result_1;

    if (rpc::is_root()) {
      fprintf(stderr, "Doing density estimation now...\n");
    }

    fx_timer_start(module, "kde_1");
    thor::RpcDualTree<FdKde, SOLVER_TYPE<FdKde> >(
        fx_submodule(module, "gnp", "kde_1"), 200,
        *param, &tree, &tree, &results, &global_result_1);
    fx_timer_stop(module, "kde_1");
    results.ResetElements();

    String kernel_type = fx_param_str_req(module, "kde/kernel");
    if (kernel_type == "gauss_star") {
      if (rpc::is_root()) {
        fprintf(stderr, "Doing density estimation with second kernel for LSCV...\n");
      }
      // Gaussian convolution kernel.  Run again at the modified bandwidth.
      GlobalResult global_result_2;
      
      param->kernel.Init(sqrt(param->kernel.bandwidth_sq() * 2));
      
      fx_timer_start(module, "kde_2");
      thor::RpcDualTree<FdKde, SOLVER_TYPE<FdKde> >(
          fx_submodule(module, "gnp", "kde_2"), 200,
          *param, &tree, &tree, &results, &global_result_2);
      fx_timer_stop(module, "kde_2");
      results.ResetElements();
      
      if (rpc::is_root()) {
        DRange lscv_value;
        
        lscv_value.lo = global_result_2.sum_density.lo
            - 2 * global_result_1.sum_density.hi;
        lscv_value.hi = global_result_2.sum_density.hi
            - 2 * global_result_1.sum_density.lo;
        
        fx_format_result(module, "lscv/value", "%g", lscv_value.mid());
        fx_format_result(module, "lscv/error", "%g", lscv_value.width() / 2);
        fx_format_result(module, "lscv/lo", "%g", lscv_value.lo);
        fx_format_result(module, "lscv/hi", "%g", lscv_value.hi);
      }
    }

    delete param;
    
    rpc::Done();
  }
};

void KdeMain(datanode *module) {
  String kernel = fx_param_str(module, "kde/kernel", "gauss");

  if (kernel == "gauss" || kernel == "gauss_star") {
    FdKde<GaussianKernel>::DoKde(module);
  } else if (kernel == "epan") {
    FdKde<EpanKernel>::DoKde(module);
  } else {
    FATAL("Unsupported kernel: '%s'", kernel.c_str());
  }
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  KdeMain(fx_root);
  
  fx_done();
}

