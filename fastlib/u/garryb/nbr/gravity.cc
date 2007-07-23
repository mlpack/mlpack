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
    double theta;
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
      theta = fx_param_double_req(datanode, "theta");
      //theta_factor = math::Sqr(1.0 + theta);
      //theta_factor = math::Sqr(1.0 / (1.0 / theta + sqrt(3)));
      theta_factor = math::Sqr(theta + 1);
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

  struct CombinedStat {
   public:
    double diagsq;
    Vector centroid;

    OT_DEF(CombinedStat) {
      OT_MY_OBJECT(diagsq);
      OT_MY_OBJECT(centroid);
    }

   public:
    void Init(const Param& param) {
      centroid.Init(param.dim);
    }
    void Reset(const Param& param) {
      centroid.SetZero();
    }
    void Accumulate(const Param& param, const QPoint& point) {
      la::AddTo(point.vec(), &centroid);
    }
    void Accumulate(const Param& param,
        const CombinedStat& stat, const Bound& bound, index_t n) {
      la::AddTo(stat.centroid, &centroid);
    }
    void Postprocess(const Param& param, const Bound& bound, index_t n) {
      diagsq = bound.MaxDistanceSqToBound(bound);
      la::Scale(1.0 / n, &centroid);
    }
  };

  typedef SpNode<Bound, CombinedStat> RNode;
  typedef SpNode<Bound, CombinedStat> QNode;

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
      /*double dhi = r_node.bound().MaxDistanceSqToPoint(q_point.vec());
      double dlo = r_node.bound().MinDistanceSqToPoint(q_point.vec());
      if (dhi > dlo * param.theta_factor) {
        force = 0;
        return true;
      } else {
        q_result->force += param.Force((dhi + dlo) / 2);
        return false;
      }*/
      force = 0;
      return true;
    }

    void VisitPair(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RPoint& r_point, index_t r_index) {
      //double distsq = la::DistanceSqEuclidean(
      //    q_point.vec(), r_point.vec());
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
      double distsq_lo = q_node.bound().MinDistanceSqToBound(r_node.bound());
      double distsq_hi = q_node.bound().MaxDistanceSqToBound(r_node.bound());
      //double distsq_mid = q_node.bound().MidDistanceSqToBound(r_node.bound());
      //double diagsq = q_node.stat().diagsq + r_node.stat().diagsq;
      // (sqrt(distsq_hi) - sqrt(distsq_lo)) / sqrt(distsq_lo) < theta
      // sqrt(distsq_hi) / sqrt(distsq_lo) - 1 < theta
      // sqrt(distsq_hi) / sqrt(distsq_lo) < theta + 1
      // sqrt(distsq_hi) / sqrt(distsq_lo) < theta + 1
      // distsq_hi / distsq_lo < (theta + 1)^2
      bool should_explore = (distsq_hi >= distsq_lo * param.theta_factor);
      if (!should_explore) {
        double distsq_centroid = la::DistanceSqEuclidean(q_node.stat().centroid, r_node.stat().centroid);
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
