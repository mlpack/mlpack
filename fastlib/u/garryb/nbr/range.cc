#include "fastlib/fastlib.h"
#include "spbounds.h"
#include "gnp.h"
#include "dfs.h"
#include "nbr_utils.h"

/**
 * An N-Body-Reduce problem.
 */
class Range {
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
    index_t dim;
    index_t count;
    double h_orig;
    double h_sq;

    OT_DEF(Param) {
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(count);
      OT_MY_OBJECT(h_sq);
      OT_MY_OBJECT(h_orig);
    }

   public:
    void Copy(const Param& other) {
      dim = other.dim;
      count = other.count;
      h_sq = other.h_sq;
      h_orig = other.h_orig;
    }

    /**
     * Initialize parameters from a data node (Req NBR).
     */
    void Init(datanode *datanode) {
      dim = -1;
      count = -1;
      h_orig = fx_param_double_req(datanode, "h");
      h_sq = h_orig * h_orig;
    }

    void BootstrapMonochromatic(QPoint* point, index_t count_in) {
      dim = point->vec().length();
      count = count_in;
    }
  };

  typedef BlankStat QStat;
  typedef BlankStat RStat;

  typedef SpNode<Bound, BlankStat> RNode;
  typedef SpNode<Bound, BlankStat> QNode;

  typedef BlankDelta Delta;

  struct GlobalResult {
   public:
    uint64 count;

   public:
    void Init(const Param& param) {
      count = 0;
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
      count += other.count;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
      fx_format_result(datanode, "per_point_avg", "%g",
          count / double(param.count));
      fx_format_result(datanode, "per_pair_avg", "%g",
          count / (double(param.count) * double(param.count)));
    }

   public:
    void Add(index_t q_count, index_t r_count) {
      count += uint64(q_count) * r_count;
    }
  };

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
   public:
    int count;

    OT_DEF(QPostponed) {
      OT_MY_OBJECT(count);
    }

   public:
    void Init(const Param& param) {
      count = 0;
    }

    void Reset(const Param& param) {
      count = 0;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      count += other.count;
    }
  };


  // rho
  struct QResult {
   public:
    index_t count;

    OT_DEF(QResult) {
      OT_MY_OBJECT(count);
    }

   public:
    void Init(const Param& param) {
      count = 0;
    }

    void Postprocess(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RNode& r_root) {}
    void ApplyPostponed(const Param& param,
        const QPostponed& postponed,
        const QPoint& q_point,
        index_t q_index) {
      count += postponed.count;
    }
  };

  typedef BlankQMassResult QMassResult;

  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct PairVisitor {
   public:
    index_t count;

   public:
    void Init(const Param& param) {}

    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      double distance_sq_lo =
          r_node.bound().MinDistanceSqToPoint(q_point.vec());
      if (unlikely(distance_sq_lo > param.h_sq)) {
        return false;
      }
      double distance_sq_hi =
          r_node.bound().MaxDistanceSqToPoint(q_point.vec());
      if (unlikely(distance_sq_hi <= param.h_sq)) {
        q_result->count += r_node.count();
        global_result->count += r_node.count();
        return false;
      }
      count = 0;
      return true;
    }

    void VisitPair(const Param& param,
        const QPoint& q_point, index_t q_index,
        const RPoint& r_point, index_t r_index) {
      double trial_distance_sq = la::DistanceSqEuclidean(
          q_point.vec(), r_point.vec());
      if (trial_distance_sq <= param.h_sq) {
        count++;
      }
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q_point,
        index_t q_index,
        const RNode& r_node,
        const QMassResult& unapplied_mass_results,
        QResult* q_result,
        GlobalResult* global_result) {
      q_result->count += count;
      global_result->count += count;
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
      double distance_sq_lo =
          q_node.bound().MinDistanceSqToBound(r_node.bound());
      if (unlikely(distance_sq_lo > param.h_sq)) {
        return false;
      }
      double distance_sq_hi =
          q_node.bound().MaxDistanceSqToBound(r_node.bound());
      if (unlikely(distance_sq_hi <= param.h_sq)) {
        q_postponed->count += r_node.count();
        global_result->Add(q_node.count(), r_node.count());
        return false;
      }
      return true;
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

#ifdef USE_RPC
  nbr_utils::RpcMonochromaticDualTreeMain<Range, DualTreeDepthFirst<Range> >(
      fx_root, "range");      
#else
  nbr_utils::MonochromaticDualTreeMain<Range, DualTreeDepthFirst<Range> >(
      fx_root, "range");
#endif
  
  fx_done();
}
