#include "thor/thor.h"
#include "fastlib/fastlib.h"

/**
 * An N-Body-Reduce problem.
 */
class Allnn {
 public:
  /** The bounding type. Required by THOR. */
  typedef DHrectBound<2> Bound;

  /** The type of point in use. Required by THOR. */
  typedef ThorVectorPoint QPoint;
  typedef ThorVectorPoint RPoint;

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
    
    void InitPointExtras(int tag, ThorVectorPoint *point) const {}
    void SetPointExtras(int tag, index_t index, ThorVectorPoint *point) const {}
    
    void Bootstrap(int tag, index_t dim_in, index_t count) {
      dim = dim_in;
    }
  };

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

    void VisitPair(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RPoint& r_point, index_t r_index) {
      double trial_distance_sq = la::DistanceSqEuclidean(
          q_point.vec(), r_point.vec());
      if (unlikely(trial_distance_sq <= distance_sq)) {
        // TODO: Is this a hack?
        if (likely(trial_distance_sq != 0)) {
          neighbor_i = r_index;
          distance_sq = trial_distance_sq;
        }
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
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
      double distance_sq_lo =
          q_node.bound().MinDistanceSq(r_node.bound());
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

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  thor_utils::MonochromaticDualTreeMain<Allnn, DualTreeDepthFirst<Allnn> >(
      fx_root, "allnn");
  
  fx_done();
}
