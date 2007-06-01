#include "fastlib/fastlib.h"
#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"
#include "nbr_utils.h"

/**
 * An N-Body-Reduce problem.
 */
class Allnn {
 public:
  /** The bounding type. Required by NBR. */
  typedef SpHrectBound<2> Bound;
  /** The type of point in use. Required by NBR. */

  typedef SpVectorPoint QPoint;
  typedef SpVectorPoint RPoint;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by N-Body Reduce.
   */
  struct Param {
   public:
    /** The dimensionality of the data sets. */
    index_t dim;

    OT_DEF(Param) {
      OT_MY_OBJECT(dim);
    }

   public:
    void Copy(const Param& other) {
      dim = other.dim;
    }
   
    /**
     * Initialize parameters from a data node (Req NBR).
     */
    void Init(datanode *datanode) {
      dim = -1;
    }
    
    void BootstrapMonochromatic(QPoint* point, index_t count) {
      dim = point->vec().length();
    }

    void BootstrapQueries(QPoint* point, index_t count) {
      dim = point->vec().length();
    }

    void BootstrapReferences(RPoint* point, index_t count) {
      dim = point->vec().length();
    }
  };

  typedef BlankStat QStat;
  typedef BlankStat RStat;

  typedef SpNode<Bound, BlankStat> RNode;
  typedef SpNode<Bound, BlankStat> QNode;

  typedef BlankQPostponed QPostponed;
  typedef BlankDelta Delta;
  
  typedef BlankGlobalResult GlobalResult;

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
    void Init(const Param& param) {
      distance_sq = DBL_MAX;
      neighbor_i = -1;
    }

    void Postprocess(const Param& param,
        const QPoint& q_point,
        const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q_point) {}
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
      distance_sq_hi = 0;
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
        const QPoint& q_point,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      /* ignore horizontal join operator */
      distance_sq = q_result->distance_sq;
      neighbor_i = q_result->neighbor_i;
      return r_node.bound().MinDistanceSqToPoint(q_point.vec()) <= distance_sq;
    }

    void VisitPair(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RPoint& r_point, index_t r_index) {
      double trial_distance_sq = la::DistanceSqEuclidean(
          q_point.vec(), r_point.vec()	);
      if (unlikely(trial_distance_sq <= distance_sq)) {
        // TODO: Is this a hack?
        if (likely(trial_distance_sq != 0)) {
          distance_sq = trial_distance_sq;
          neighbor_i = r_index;
        }
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
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
    static bool ConsiderPairIntrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node,
        Delta* delta,
        GlobalResult* global_result, QPostponed* q_postponed) {
      return true;
    }

    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      double distance_sq_lo =
          q_node.bound().MinDistanceSqToBound(r_node.bound());
      return distance_sq_lo <= q_mass_result.distance_sq_hi;
    }

    static bool ConsiderQueryTermination(const Param& param,
        const QNode& q_node,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }

    /**
     * Computes a heuristic for how early a computation should occur -- smaller
     * values are earlier.
     */
    static double Heuristic(const Param& param,
        const QNode& q_node,  const RNode& r_node, const Delta& delta) {
      return q_node.bound().MinDistanceSqToBound(r_node.bound());
    }
  };
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

#ifdef USE_MPI  
  MPI_Init(&argc, &argv);
  nbr_utils::MpiDualTreeMain<Allnn, DualTreeDepthFirst<Allnn> >(
      fx_root, "allnn");
  MPI_Finalize();
#else
  nbr_utils::MonochromaticDualTreeMain<Allnn, DualTreeDepthFirst<Allnn> >(
      fx_root, "allnn");
#endif
  
  fx_done();
}
