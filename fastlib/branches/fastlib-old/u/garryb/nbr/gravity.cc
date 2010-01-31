#include "thor/thor.h"
#include "fastlib/fastlib.h"

/**
 * An N-Body-Reduce problem.
 */
class Gravity {
 public:
  /** Gravity simulators only make sense in 3 dimensions. */
  enum { DIM = 3 };

  /** The bounding type. Required by THOR. */
  typedef DHrectBound<2> Bound;
  /** The type of point in use. Required by THOR. */

  typedef ThorVectorPoint QPoint;
  typedef ThorVectorPoint RPoint;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by N-Body Reduce.
   */
  struct Param {
   public:
    double theta;
    double theta_factor;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(theta);
      OT_MY_OBJECT(theta_factor);
    }

   public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *datanode) {
      theta = fx_param_double_req(datanode, "theta");
      //theta_factor = math::Sqr(1.0 + theta);
      //theta_factor = math::Sqr(1.0 / (1.0 / theta + sqrt(3)));
      theta_factor = math::Sqr(theta + 1);
    }

    void InitPointExtras(int tag, QPoint* point) {
    }
    void SetPointExtras(int tag, index_t index, QPoint* point) {
    }
    void Bootstrap(int tag, index_t dim_in, index_t count) {
      DEBUG_ASSERT(dim_in == DIM);
    }
    double Force(double distsq) const {
      return 1.0/distsq;
    }
  };

  struct CombinedStat {
   public:
    double diagsq;
    double centroid[DIM];

    OT_DEF_BASIC(CombinedStat) {
      OT_MY_OBJECT(diagsq);
      OT_MY_ARRAY(centroid);
    }

   public:
    void Init(const Param& param) {
    }
    void Reset(const Param& param) {
      for (int i = 0; i < DIM; i++) {
        centroid[i] = 0;
      }
    }
    void Accumulate(const Param& param, const QPoint& point) {
      la::AddTo(DIM, point.vec().ptr(), centroid);
    }
    void Accumulate(const Param& param,
        const CombinedStat& stat, const Bound& bound, index_t n) {
      la::AddTo(DIM, stat.centroid, centroid);
    }
    void Postprocess(const Param& param, const Bound& bound, index_t n) {
      diagsq = bound.MaxDistanceSq(bound);
      la::Scale(DIM, 1.0 / n, centroid);
    }
  };

  typedef ThorNode<Bound, CombinedStat> RNode;
  typedef ThorNode<Bound, CombinedStat> QNode;

  typedef BlankDelta Delta;
  
  typedef BlankGlobalResult GlobalResult;

  struct QPostponed {
   public:
    double force;

    OT_DEF_BASIC(QPostponed) {
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

    OT_DEF_BASIC(QResult) {
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

  struct QSummaryResult {
   public:
    void Init(const Param& param) {}
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {}
    void ApplySummaryResult(const Param& param, const QSummaryResult& summary_result) {}
    void StartReaccumulate(const Param& param, const QNode& q_node) {}
    void Accumulate(const Param& param, const QResult& result) {}
    void Accumulate(const Param& param,
        const QSummaryResult& result, index_t n_points) {}
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
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {
      double distsq_lo = r_node.bound().MinDistanceSq(q_point.vec());
      double distsq_hi = r_node.bound().MaxDistanceSq(q_point.vec());
      bool should_explore = (distsq_hi >= distsq_lo * param.theta_factor);
      force = 0;
      if (!should_explore) {
        double distsq_centroid = la::DistanceSqEuclidean(
            DIM, q_point.vec().ptr(), r_node.stat().centroid);
        q_result->force += param.Force(distsq_centroid);
      }
      return should_explore;
    }

    void VisitPair(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RPoint& r_point, index_t r_index) {
      const double *a = q_point.vec().ptr();
      const double *b = r_point.vec().ptr();
      double x = a[0]-b[0];
      double y = a[1]-b[1];
      double z = a[2]-b[2];
      double distsq = x*x + y*y + z*z;
      if (likely(distsq != 0)) {
        force += param.Force(distsq);
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
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
      double distsq_lo = r_node.bound().MinDistanceSq(q_node.bound());
      double distsq_hi = r_node.bound().MaxDistanceSq(q_node.bound());
      bool should_explore = (distsq_hi >= distsq_lo * param.theta_factor);
      if (!should_explore) {
        double distsq_centroid = la::DistanceSqEuclidean(
            DIM, q_node.stat().centroid, r_node.stat().centroid);
        q_postponed->force += r_node.count() * param.Force(distsq_centroid);
      }
      return should_explore;
    }

    static bool ConsiderPairExtrinsic(const Param& param,
        const QNode& q_node, const RNode& r_node, const Delta& delta,
        const QSummaryResult& q_summary_result, const GlobalResult& global_result,
        QPostponed* q_postponed) {
      return true;
    }

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
      return 0;
    }
  };
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  thor_utils::MonochromaticDualTreeMain<Gravity, DualTreeDepthFirst<Gravity> >(
      fx_root, "gravity");
  
  fx_done();
}
