#include "fastlib/fastlib.h"
#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"

/**
 * An N-Body-Reduce problem.
 */
class Allnn {
 public:
  /** The bounding type. Required by NBR. */
  typedef SpHrectBound<2> Bound;
  /** The type of point in use. Required by NBR. */
  typedef Vector Point;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by N-Body Reduce.
   */
  struct Param {
   public:
    /** The dimensionality of the data sets. */
    index_t dim;
    double max_dist;

    OT_DEF(Param) {
      OT_MY_OBJECT(dim);
    }

   public:
    /**
     * Initialize parameters from a data node (Req NBR).
     */
    void Init(datanode *datanode, const Matrix& q_matrix,
        const Matrix& r_matrix) {
      dim = q_matrix.n_rows();
      max_dist = 1.0e8;
    }
  };

  typedef BlankPointInfo QPointInfo;
  typedef BlankPointInfo RPointInfo;
  
  typedef BlankStat<Param, Point, Bound, QPointInfo> QStat;
  typedef BlankStat<Param, Point, Bound, RPointInfo> RStat;

  typedef SpNode<Bound, RStat> RNode;
  typedef SpNode<Bound, QStat> QNode;

  typedef BlankQPostponed<Param> QPostponed;
  typedef BlankDelta<Param> Delta;

  // rho
  struct QResult {
   public:
    double distance_sq;
    index_t neighbor_i;

    OT_DEF(QResult) {
      OT_MY_OBJECT(distance_sq);
      OT_MY_OBJECT(neighbor_i);
    }

   public:
    void Init(const Param& param,
        const Vector& q_point, const QPointInfo& q_info,
        const RNode& r_root) {
      distance_sq = param.max_dist;
      neighbor_i = -1;
    }

    void Postprocess(const Param& param,
        const Vector& q_point, const QPointInfo& q_info,
        const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const Vector& q_point) {}
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
    MinMaxVal<double> distance_sq_hi;

    OT_DEF(QMassResult) {
      OT_MY_OBJECT(distance_sq_hi);
    }

   public:
    void Init(const Param& param) {
      distance_sq_hi = DBL_MAX;
    }

    void ApplyDelta(const Param& param, const Delta& delta) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {}

    void ApplyMassResult(const Param& param, const QMassResult& mass_result) {
      distance_sq_hi.MinWith(mass_result.distance_sq_hi);
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      distance_sq_hi = -DBL_MAX;
    }

    void Accumulate(const Param& param, const QResult& result) {
      distance_sq_hi.MaxWith(result.distance_sq);
    }

    void Accumulate(const Param& param,
        const QMassResult& result, index_t n_points) {
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
        const Vector& q_point,
        const QPointInfo& q_info,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      /* ignore horizontal join operator */
      distance_sq = q_result->distance_sq;
      neighbor_i = q_result->neighbor_i;
      return r_node.bound().MinDistanceSqToPoint(q_point) < distance_sq;
    }

    void VisitPair(const Param& param,
        const Vector& q_point, const QPointInfo& q_info, index_t q_index,
        const Vector& r_point, const RPointInfo& r_info, index_t r_index) {
      double trial_distance_sq = la::DistanceSqEuclidean(q_point, r_point);
      if (unlikely(trial_distance_sq < distance_sq)) {
        // TODO: Is this a hack?
        if (likely(trial_distance_sq  != 0)) {
          distance_sq = trial_distance_sq;
          neighbor_i = r_index;
        }
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const Vector& q_point,
        const QPointInfo& q_info,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      q_result->distance_sq = distance_sq;
      q_result->neighbor_i = neighbor_i;
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
      return true;
    }

    static bool ConsiderPairExtrinsic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta,
        const QMassResult& q_mass_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
      double distance_sq_lo =
          q_node.bound().MinDistanceSqToBound(r_node.bound());
      return distance_sq_lo <= q_mass_result.distance_sq_hi;
    }

    static bool ConsiderQueryTermination(
        const Param& param,
        const QNode& q_node,
        const QMassResult& q_mass_result,
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
  
  DualTreeDepthFirst<Allnn::GNP> dfs;
  dfs.Init(fx_root);
  dfs.Begin();
  
  fx_done();
}
