#ifndef THOR_KDE_H
#define THOR_KDE_H

#include "fastlib/fastlib_int.h"
#include "thor/thor.h"
#include "u/dongryel/series_expansion/kernel_aux.h"


/**
 * THOR-based KDE
 */
template<typename TKernel, typename TKernelAux>
class ThorKde {

 public:

  /** the bounding type which is required by THOR */
  typedef DHrectBound<2> Bound;

  /** parameter class */
  class Param {
  public:
      
    /** the dimensionality of the datasets */
    index_t dimension_;
    
    /** number of query points */
    index_t query_count_;

    /** number of reference points */
    index_t reference_count_;
   
    /** the global relative error allowed */
    double relative_error_;
    
    /** the bandwidth */
    double bandwidth_;

    /** multiply the unnormalized sum by this to get the density estimate */
    double mul_constant_;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(dimension_);
      OT_MY_OBJECT(reference_count_);
      OT_MY_OBJECT(query_count_);
      OT_MY_OBJECT(relative_error_);
      OT_MY_OBJECT(bandwidth_);
      OT_MY_OBJECT(mul_constant_);
    }

    /**
     * Initializes parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      
      // get bandwidth and relative error
      bandwidth_ = fx_param_double_req(module, "bandwidth");
      relative_error_ = fx_param_double(module, "tau", 0.1);

      // temporarily initialize these to -1's
      dimension_ = reference_count_ = query_count_ = -1;
    }

    void FinalizeInit(datanode *module, int dimension, int query_count,
		      int reference_count) {
      dimension_ = dimension;
      query_count_ = query_count;
      reference_count_ = reference_count;

      /*
      // initialize the series expansion object
      if(fx_param_exists(module, "multiplicative_expansion")) {
      if(dimension_ <= 2) {
      sea_.Init(fx_param_int(module, "order", 5), qset_.n_rows());
      }
      else if(dimension_ <= 3) {
      sea_.Init(fx_param_int(module, "order", 1), qset_.n_rows());
      }
      else {
      sea_.Init(fx_param_int(module, "order", 0), qset_n_rows());
      }
      }
      else {
      if(dimension_ <= 2) {
      sea_.Init(fx_param_int(module, "order", 7), qset_.n_rows());
      }
      else if(dimension_ <= 3) {
      sea_.Init(fx_param_int(module, "order", 3), qset_.n_rows());
      }
      else if(dimension_ <= 5) {
      sea_.Init(fx_param_int(module, "order", 1), qset_n_rows());
      }
      else {
      sea_.Init(fx_param_int(module, "order", 0), qset_n_rows());
      }
      }
      */
    }
  };

  /** the type of each KDE point */
  class ThorKdePoint {
  public:
    
    /** the point's position */
    Vector v_;
    
    /** the weight for each reference point */
    double weight_;
    
    OT_DEF(ThorKdePoint) {
      OT_MY_OBJECT(v_);
      OT_MY_OBJECT(weight_);
    }

  public:

    /** getters for the vector so that the tree-builder can access it */
    const Vector& vec() const { return v_; }
    Vector& vec() { return v_; }

    /** initializes all memory for a point */
    void Init(const Param& param, const DatasetInfo& schema) {
      v_.Init(schema.n_features() - 1);
      v_.SetZero();
      weight_ = 1.0;
    }

    /** 
     * sets contents assuming all space has been allocated.
     * Any attempt to allocate memory here will lead to a core dump.
     */
    void Set(const Param& param, index_t index, Vector& data) {
      Vector tmp;
      DEBUG_ASSERT(data.length() == v_.length() + 1);
      data.MakeSubvector(0, v_.length(), &tmp);
      v_.CopyValues(tmp);
      weight_ = data[data.length() - 1];
    }    
  };



 /**
  * Per-node bottom-up statistic for both queries and references.
  *
  * The statistic must be commutative and associative, thus bottom-up
  * computable.
  *
  */
 class ThorKdeStat {
    
 public:
    
   /**
    * far field expansion created by the reference points in this node.
    */
   typename TKernelAux::TFarFieldExpansion far_field_expansion_;
    
   /** local expansion stored in this node.
    */
   typename TKernelAux::TLocalExpansion local_expansion_;

   OT_DEF(ThorKdeStat) {
     OT_MY_OBJECT(far_field_expansion_);
     OT_MY_OBJECT(local_expansion_);
   }

   /**
    * Initialize to a default zero value, as if no data is seen (Req THOR).
    *
    * This is the only method in which memory allocation can occur.
    */
 public:
   void Init(const Param& param) {
     far_field_expansion_.Init(param.bandwidth_, NULL);
     local_expansion_.Init(param.bandwidth_, NULL);
   }

   /**
    * Accumulate data from a single point (Req THOR).
    */
   void Accumulate(const Param& param, const ThorKdePoint& point) {
     far_field_expansion_.Accumulate(point.vec(), point.weight_, 0);
   }

   /**
    * Accumulate data from one of your children (Req THOR).
    */
   void Accumulate(const Param& param, const ThorKdeStat& child_stat, 
		   const Bound& bound, index_t child_n_points) {
     far_field_expansion_.
       TranslateFromFarField(child_stat.far_field_expansion_);
   }
    
   /**
    * Finish accumulating data; for instance, for mean, divide by the
    * number of points.
    */
 public:

   void Postprocess(const Param& param, const Bound&bound, index_t n) {      
   }
 };

 typedef ThorKdePoint QPoint;
 typedef ThorKdePoint RPoint;


 /** query stat */
 typedef ThorKdeStat QStat;
  
 /** reference stat */
 typedef ThorKdeStat RStat;

 /** query node */
 typedef ThorNode<Bound, QStat> QNode;

  /** reference node */
  typedef ThorNode<Bound, RStat> RNode;

  /**
   * Coarse result on a region.
   */
  class Delta {
   public:

    /** Density update to apply to children's bound. Similar for _neg. */

    OT_DEF_BASIC(Delta) {
    }

   public:
    void Init(const Param& param) {
    }
  };

  /** coarse result on a region */
  class QPostponed {
   public:

    OT_DEF_BASIC(QPostponed) {
    }

   public:
    void Init(const Param& param) {
    }

    void Reset(const Param& param) {
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
    }
  };

  /** individual query result */
  class QResult {
   public:
    double density_;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density_);
    }

   public:
    void Init(const Param& param) {
      density_ = 0.0;
    }

    void Seed(const Param& param, const QPoint& q) {
    }

    void Postprocess(const Param& param, const QPoint& q, index_t q_index,
		     const RNode& r_root) {
    }

    void ApplyPostponed(const Param& param, const QPostponed& postponed,
			const QPoint& q, index_t q_index) {
    }
  };

  class QSummaryResult {
   public:

    OT_DEF_BASIC(QSummaryResult) {
    }

   public:
    void Init(const Param& param) {

    }

    void Seed(const Param& param, const QNode& q_node) {

    }

    // Why does this have a q_node?  RR
    void StartReaccumulate(const Param& param, const QNode& q_node) {

    }

    void Accumulate(const Param& param, const QResult& result) {

    }

    void Accumulate(const Param& param,
		    const QSummaryResult& result, index_t n_points) {

    }

    void FinishReaccumulate(const Param& param,
        const QNode& q_node) {
      /* no post-processing steps necessary */
    }

    /** horizontal join operator */
    void ApplySummaryResult(const Param& param,
        const QSummaryResult& summary_result) {

    }

    void ApplyDelta(const Param& param,
        const Delta& delta) {

    }

    void ApplyPostponed(const Param& param,
        const QPostponed& postponed, const QNode& q_node) {

    }
  };

  /**
   * A simple postprocess-step global result.
   */
  class GlobalResult {
   public:
    
    OT_DEF_BASIC(GlobalResult) {
    }

   public:
    void Init(const Param& param) {
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
    }
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
    }
    void ApplyResult(const Param& param,
        const QPoint& q_point, index_t q_i,
        const QResult& result) {
    }
  };

    /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct PairVisitor {
   public:
    double density_pos;
    double density_neg;

   public:
    void Init(const Param& param) {}

    // notes
    // - this function must assume that global_result is incomplete (which is
    // reasonable in allnn)
    bool StartVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node,
	const Delta& delta,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {

      return true;
    }

    void VisitPair(const Param& param,
        const QPoint& q, index_t q_index,
        const RPoint& r, index_t r_index) {
    }

    void FinishVisitingQueryPoint(const Param& param,
        const QPoint& q, index_t q_index,
        const RNode& r_node,
        const QSummaryResult& unapplied_summary_results,
        QResult* q_result,
        GlobalResult* global_result) {

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
	const Delta& parent_delta,
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
        const QSummaryResult& q_summary_result,
        const GlobalResult& global_result,
        QPostponed* q_postponed) {
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
     * Computes a heuristic for how early a computation should occur
     * -- smaller values are earlier.
     */
    static double Heuristic(
        const Param& param,
        const QNode& q_node,
        const RNode& r_node,
        const Delta& delta) {
      return r_node.bound().MinToMidSq(q_node.bound());
    }
  };

  // functions
  
  /** read datasets, build trees */
  void Init(datanode *module) {

    // I don't quite understand what these mean, since I copied and pasted
    // from an example code.
    double results_megs = fx_param_double(module, "results/megs", 1000);

    rpc::Init();
    
    if (!rpc::is_root()) {
      fx_silence();
    }

    // read reference dataset
    fx_timer_start(module, "read_datasets");
    r_points_cache_ = new DistributedCache();
    parameters_.reference_count_ = 
      thor::ReadPoints<RPoint>(parameters_, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
			       fx_submodule(module, "r", "r"),
			       r_points_cache_);

    // read the query dataset if present
    if(fx_param_exists(module, "q")) {
      q_points_cache_ = new DistributedCache();
      parameters_.query_count_ = thor::ReadPoints<QPoint>
	(parameters_, DATA_CHANNEL + 2, DATA_CHANNEL + 3,
	 fx_submodule(module, "query", "query"), q_points_cache_);
    } 
    else {
      q_points_cache_ = r_points_cache_;
      parameters_.query_count_ = parameters_.reference_count_;
    }
    fx_timer_stop(module, "read_datasets");

    // construct trees
    fx_timer_start(module, "tree_construction");
    r_tree_ = new ThorTree<Param, RPoint, RNode>();
    thor::CreateKdTree<RPoint, RNode>(parameters_, DATA_CHANNEL + 4, 
				      DATA_CHANNEL + 5,
				      fx_submodule(module, "r_tree", "r_tree"),
				      parameters_.reference_count_, 
				      r_points_cache_, r_tree_);
    if (fx_param_exists(module, "q")) {
      q_tree_ = new ThorTree<Param, QPoint, QNode>();
      thor::CreateKdTree<QPoint, QNode>
	(parameters_, DATA_CHANNEL + 6, DATA_CHANNEL + 7, 
	 fx_submodule(module, "q_tree", "q_tree"), parameters_.query_count_,
	 q_points_cache_, q_tree_);
    } 
    else {
      q_tree_ = r_tree_;
    }
    fx_timer_stop(module, "tree_construction");

    // set up the cache holding query results
    QResult default_result;
    default_result.Init(parameters_);
    q_tree_->CreateResultCache(Q_RESULTS_CHANNEL, default_result,
			       results_megs, &q_results_);    
  }

  /** distributed cache for storing query results */
  DistributedCache q_results_;

  /** distributed cache for query points */
  DistributedCache *q_points_cache_;

  /** thor tree on query points */
  ThorTree<Param, QPoint, QNode> *q_tree_;
  
  /** distributed cache for reference points */
  DistributedCache *r_points_cache_;

  /** thor tree on reference points */
  ThorTree<Param, RPoint, RNode> *r_tree_;

  /** global parameter collection */
  Param parameters_;

  /** global results */
  GlobalResult global_result_;

  /** data channel */
  static const int DATA_CHANNEL = 110;

  /** query results channel */
  static const int Q_RESULTS_CHANNEL = 120;

  /** GNP channel ? */
  static const int GNP_CHANNEL = 200;

  /** the kernel in use */
  TKernel kernel_;

  /** precomputed constants for series expansion */
  static SeriesExpansionAux sea_;
};

#endif
