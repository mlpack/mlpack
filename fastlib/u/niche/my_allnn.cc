/**
 * @file allnn.cc
 *
 * A multithreaded and cluster-parallel nearest neighbors finder.
 *
 * TODO: Currently doesn't output anything.
 */

#include "thor/thor.h"
#include "fastlib/fastlib.h"

/**
 * An N-Body-Reduce problem.
 */
class Allnn {
 public:
  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by THOR.
   */
  struct Param {
   public:
    /** The dimensionality of the data sets. */
    index_t dim;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(dim);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *datanode) {
      dim = -1;
    }

    void SetDimensions(index_t vector_dimension, index_t n_points) {
      dim = vector_dimension;
    }
  };

  /** The bounding type. Required by THOR. */
  typedef DHrectBound<2> Bound;

  /** The type of point in use. Required by THOR. */
  typedef ThorVectorPoint QPoint;
  typedef ThorVectorPoint RPoint;

  typedef BlankStat QStat;
  typedef BlankStat RStat;

  typedef ThorNode<Bound, BlankStat> RNode;
  typedef ThorNode<Bound, BlankStat> QNode;

  typedef BlankQPostponed QPostponed;
  typedef BlankDelta Delta;
  
  typedef BlankGlobalResult GlobalResult;

  // rho
  struct QResult {
   public:
    double distance_sq;
    index_t neighbor_i;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(distance_sq);
      OT_MY_OBJECT(neighbor_i);
    }

   public:
    void Init(const Param& param) {
      distance_sq = DBL_MAX;
      neighbor_i = -1;
    }

    void Postprocess(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q_point,
        index_t q_index) {}
  };

  struct QSummaryResult {
   public:
    MinMaxVal<double> distance_sq_hi;

    OT_DEF_BASIC(QSummaryResult) {
      OT_MY_OBJECT(distance_sq_hi);
    }

   public:
    void Init(const Param& param) {
      distance_sq_hi = DBL_MAX;
    }

    void ApplyDelta(const Param& param, const Delta& delta) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {}

    void ApplySummaryResult(const Param& param, const QSummaryResult& summary_result) {
      distance_sq_hi.MinWith(summary_result.distance_sq_hi);
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      distance_sq_hi = 0;
    }

    void Accumulate(const Param& param, const QResult& result) {
      distance_sq_hi.MaxWith(result.distance_sq);
    }

    void Accumulate(const Param& param,
        const QSummaryResult& result, index_t n_points) {
      distance_sq_hi.MaxWith(result.distance_sq_hi);
    }

    void FinishReaccumulate(const Param& param, const QNode& q_node) {}
  };

  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct PairVisitor {
   public:
    double distance_sq;
    index_t neighbor_i;

   public:
    void Init(const Param& param) {}

    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      /* ignore horizontal join operator */
      distance_sq = q_result->distance_sq;
      neighbor_i = q_result->neighbor_i;
      return r_node.bound().MinDistanceSq(q_point.vec()) <= distance_sq;
    }

    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        QResult* q_result,
        GlobalResult* global_result) {
      distance_sq = q_result->distance_sq;
      neighbor_i = q_result->neighbor_i;
      return true;
    }

    void VisitPair(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RPoint& r_point, index_t r_index) {
      double trial_distance_sq = la::DistanceSqEuclidean(
          q_point.vec(), r_point.vec());
      if (unlikely(trial_distance_sq <= distance_sq)) {
        // TODO: Is this really indicative of q != r?
        if (likely(trial_distance_sq != 0)) {
          neighbor_i = r_index;
          distance_sq = trial_distance_sq;
        }
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RNode& r_node, const QSummaryResult& unapplied_summary_results,
        QResult* q_result, GlobalResult* global_result) {
      q_result->distance_sq = distance_sq;
      q_result->neighbor_i = neighbor_i;
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RNode& r_node,
        QResult* q_result, GlobalResult* global_result) {
      q_result->distance_sq = distance_sq;
      q_result->neighbor_i = neighbor_i;
    }
  };

  class Algorithm {
   public:
    /**
     * Calculates a delta and intrinsic pruning.
     */
    static bool ConsiderPairIntrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node,
        Delta* delta,
        GlobalResult* global_result, QPostponed* q_postponed) {
      return true;
    }

    /**
     * Attempts extrinsic pruning.
     */
    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QSummaryResult& q_summary_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      double distance_sq_lo = q_node.bound().MinDistanceSq(r_node.bound());
      return distance_sq_lo <= q_summary_result.distance_sq_hi;
    }

    /**
     * Attempts termination pruning.
     */
    static bool ConsiderQueryTermination(const Param& param,
        const QNode& q_node,
        const QSummaryResult& q_summary_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }

    /**
     * Computes a heuristic for how early a computation should occur -- smaller
     * values are earlier.
     */
    static double Heuristic(const Param& param,
        const QNode& q_node,  const RNode& r_node, const Delta& delta) {
      return r_node.bound().MinToMidSq(q_node.bound());
    }
  };
};





template<typename GNP, typename Solver>
void MyMonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  const int DATA_CHANNEL = 110;
  const int Q_RESULTS_CHANNEL = 120;
  const int GNP_CHANNEL = 200;
  double results_megs = fx_param_double(module, "results/megs", 1000);
  DistributedCache *points_cache;
  index_t n_points;
  ThorTree<typename GNP::Param, typename GNP::QPoint, typename GNP::QNode> tree;
  DistributedCache q_results;
  typename GNP::Param param;

  rpc::Init();

  fx_submodule(module, NULL, "io"); // influnce output order

  param.Init(fx_submodule(module, gnp_name, gnp_name));

  fx_timer_start(module, "read");
  points_cache = new DistributedCache();
  n_points = thor::ReadPoints<typename GNP::QPoint>(
      param, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
      fx_submodule(module, "data", "data"), points_cache);
  fx_timer_stop(module, "read");

  typename GNP::QPoint default_point;
  CacheArray<typename GNP::QPoint>::GetDefaultElement(
      points_cache, &default_point);
  param.SetDimensions(default_point.vec().length(), n_points);

  fx_timer_start(module, "tree");
  thor::CreateKdTree<typename GNP::QPoint, typename GNP::QNode>(
      param, DATA_CHANNEL + 2, DATA_CHANNEL + 3,
      fx_submodule(module, "tree", "tree"), n_points, points_cache, &tree);
  fx_timer_stop(module, "tree");

  typename GNP::QResult default_result;
  default_result.Init(param);
  tree.CreateResultCache(Q_RESULTS_CHANNEL, default_result,
        results_megs, &q_results);

  typename GNP::GlobalResult global_result;
  thor::RpcDualTree<GNP, Solver>(
      fx_submodule(module, "gnp", "gnp"), GNP_CHANNEL, param,
      &tree, &tree, &q_results, &global_result);

  
  // Emit the results; this needs to be folded into THOR
  Matrix classifications;
  classifications.Init(1, n_points);
  if (rpc::is_root()) {
    CacheArray<typename GNP::QResult> result_array;
    CacheArray<typename GNP::QPoint> points_array;
    result_array.Init(&q_results, BlockDevice::M_READ);
    points_array.Init(points_cache, BlockDevice::M_READ);
    CacheReadIter<typename GNP::QResult> result_iter(&result_array, 0);
    CacheReadIter<typename GNP::QPoint> points_iter(&points_array, 0);
    for (index_t i = 0; i < n_points; i++,
	   result_iter.Next(), points_iter.Next()) {
      //printf("%d", (*points_iter).index());
      printf("\ndist: %f", (*result_iter).distance_sq);
      printf("\nneighbor: %d", (*result_iter).neighbor_i);

    }
  }
  printf("\n");
    // data::Save(fx_param_str(module, "out", "out.csv"), classifications);
  
  
  
  


  rpc::Done();
}




int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  MyMonochromaticDualTreeMain<Allnn, DualTreeDepthFirst<Allnn> >(
      fx_root, "allnn");
  
  fx_done();
}
