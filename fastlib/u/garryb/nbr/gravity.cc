#include "fastlib/fastlib.h"
#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"
#include "nbr_utils.h"

/**
 * An N-Body-Reduce problem.
 */
class Gravity {
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
    /** Barnes-Hut parameter */
    double theta;
    /** Weird squared thing that I'm trying */
    double theta_factor;

    OT_DEF(Param) {
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(theta);
      OT_MY_OBJECT(theta_factor);
    }

   public:
    void Copy(const Param& other) {
      dim = other.dim;
      theta = other.theta;
      theta_factor = other.theta_factor;
    }
   
    /**
     * Initialize parameters from a data node (Req NBR).
     */
    void Init(datanode *datanode) {
      dim = -1;
      theta = fx_param_double(datanode, "theta", 0.01);
      theta_factor = (1.0 + theta) * (1.0 + theta);
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

    double Force(double distsq) const {
      return 1.0/distsq;
    }
  };

  typedef BlankStat QStat;
  typedef BlankStat RStat;

  typedef SpNode<Bound, BlankStat> RNode;
  typedef SpNode<Bound, BlankStat> QNode;

  typedef BlankDelta Delta;
  
  typedef BlankGlobalResult GlobalResult;

  struct QPostponed {
   public:
    double force;

    OT_DEF(QPostponed) {
      OT_MY_OBJECT(force);
    }

   public:
    void Init(const Param& param) {
      Reset(param);
    }

    void Reset(const Param& param) {
      force = 0;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      force += other.force;
    }
  };

  // rho
  struct QResult {
   public:
    double force;

    OT_DEF(QResult) {
      OT_MY_OBJECT(force);
    }

   public:
    void Init(const Param& param) {
      force = 0;
    }

    void Postprocess(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q_point,
        index_t q_index) {
      force += postponed.force;
    }
  };

  struct QMassResult {
   public:
    void Init(const Param& param) {}
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {}
    void ApplyMassResult(const Param& param, const QMassResult& mass_result) {}
    void StartReaccumulate(const Param& param, const QNode& q_node) {}
    void Accumulate(const Param& param, const QResult& result) {}
    void Accumulate(const Param& param,
        const QMassResult& result, index_t n_points) {}
    void FinishReaccumulate(const Param& param, const QNode& q_node) {}
  };

  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct PairVisitor {
   public:
    double force;

   public:
    void Init(const Param& param) {}

    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      force = 0;
      return true;
    }

    void VisitPair(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RPoint& r_point, index_t r_index) {
      double distsq = la::DistanceSqEuclidean(
          q_point.vec(), r_point.vec());
      force += param.Force(distsq);
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      q_result->force += force;
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
      double distsq_lo =
          q_node.bound().MinDistanceSqToBound(r_node.bound());
      double distsq_hi =
          q_node.bound().MaxDistanceSqToBound(r_node.bound());
      // (sqrt(distsq_hi) - sqrt(distsq_lo)) / sqrt(distsq_lo) <= theta
      // sqrt(distsq_hi) / sqrt(distsq_lo) - 1 <= theta
      // sqrt(distsq_hi) / sqrt(distsq_lo) <= theta + 1
      // sqrt(distsq_hi) / sqrt(distsq_lo) <= theta + 1
      // distsq_hi / distsq_lo <= (theta + 1)^2
      return distsq_hi <= distsq_lo * param.theta_factor;
    }

    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QMassResult& q_mass_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
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
      return 0;
    }
  };
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

#ifdef USE_MPI  
  MPI_Init(&argc, &argv);
  nbr_utils::MpiMonochromaticDualTreeMain<Gravity, DualTreeDepthFirst<Gravity> >(
      fx_root, "gravity");
  MPI_Finalize();
#else
  nbr_utils::MonochromaticDualTreeMain<Gravity, DualTreeDepthFirst<Gravity> >(
      fx_root, "gravity");
#endif
  
  fx_done();
}
