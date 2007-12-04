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
    
    /** the kernel in use */
    TKernel kernel_;
    
    /** precomputed constants for series expansion */
    typename TKernelAux::TSeriesExpansionAux sea_;
    
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
      OT_MY_OBJECT(kernel_);
      OT_MY_OBJECT(sea_);
      OT_MY_OBJECT(dimension_);
      OT_MY_OBJECT(reference_count_);
      OT_MY_OBJECT(query_count_);
      OT_MY_OBJECT(relative_error_);
      OT_MY_OBJECT(bandwidth_);
      OT_MY_OBJECT(mul_constant_);
    }
  public:
    
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
    
    void FinalizeInit(datanode *module, int dimension) {
      dimension_ = dimension;
      
      // initialize the kernel and compute the normalization constant to
      // multiply each density in the postprocessing step
      kernel_.Init(bandwidth_);
      mul_constant_ = 1.0 / 
	(kernel_.CalcNormConstant(dimension) * reference_count_);
      
      // initialize the series expansion object
      if(fx_param_exists(module, "multiplicative_expansion")) {
	if(dimension_ <= 2) {
	  sea_.Init(fx_param_int(module, "order", 5), dimension);
	}
	else if(dimension_ <= 3) {
	  sea_.Init(fx_param_int(module, "order", 1), dimension);
	}
	else {
	  sea_.Init(fx_param_int(module, "order", 0), dimension);
	}
      }
      else {
	if(dimension_ <= 2) {
	  sea_.Init(fx_param_int(module, "order", 7), dimension);
	}
	else if(dimension_ <= 3) {
	  sea_.Init(fx_param_int(module, "order", 3), dimension);
	}
	else if(dimension_ <= 5) {
	  sea_.Init(fx_param_int(module, "order", 1), dimension);
	}
	else {
	  sea_.Init(fx_param_int(module, "order", 0), dimension);
	}
      }
    }
  };
  
  /** 
   * the type of each KDE point - this assumes that each query and
   * each reference point is appended with a weight.
   */
  class ThorKdePoint {
  public:
    
    /** the point's position */
    Vector v_;
    
    OT_DEF(ThorKdePoint) {
      OT_MY_OBJECT(v_);
    }
    
  public:
    
    /** getters for the vector so that the tree-builder can access it */
    const Vector& vec() const { return v_; }
    Vector& vec() { return v_; }
    
    /** initializes all memory for a point */
    void Init(const Param& param, const DatasetInfo& schema) {
      v_.Init(schema.n_features());
      v_.SetZero();
    }
    
    /** 
     * sets contents assuming all space has been allocated.
     * Any attempt to allocate memory here will lead to a core dump.
     */
    void Set(const Param& param, index_t index, Vector& data) {
      Vector tmp;
      data.MakeSubvector(0, v_.length(), &tmp);
      v_.CopyValues(tmp);
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
      far_field_expansion_.Init(param.bandwidth_, &param.sea_);
      local_expansion_.Init(param.bandwidth_, &param.sea_);
    }
    
    /**
     * Accumulate data from a single point (Req THOR).
     */
    void Accumulate(const Param& param, const ThorKdePoint& point) {
      far_field_expansion_.Accumulate(point.vec(), 1, 0);
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
    void Postprocess(const Param& param, const Bound&bound, index_t n) {
      bound.CalculateMidpoint(far_field_expansion_.get_center());
      bound.CalculateMidpoint(local_expansion_.get_center());
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

    /** Density update to apply to children's bound */
    DRange d_density_;

    OT_DEF_BASIC(Delta) {
      OT_MY_OBJECT(d_density_);
    }

  public:
    void Init(const Param& param) {
    }
  };

  /** coarse result on a region */
  class QPostponed {
  public:

    DRange d_density_;
    double used_error_;
    int n_pruned_;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(d_density_);
      OT_MY_OBJECT(used_error_);
      OT_MY_OBJECT(n_pruned_);
    }

  public:
    
    /** initialize postponed information to zero */
    void Init(const Param& param) {
      d_density_.Init(0, 0);
      used_error_ = 0;
      n_pruned_ = 0;
    }

    void Reset(const Param& param) {
      d_density_.Init(0, 0);
      used_error_ = 0;
      n_pruned_ = 0;
    }

    /** accumulate postponed information passed down from above */
    void ApplyPostponed(const Param& param, const QPostponed& other) {
      d_density_ += other.d_density_;
      used_error_ += other.used_error_;
      n_pruned_ += other.n_pruned_;
    }
  };

  /** individual query result */
  class QResult {
  public:
    DRange density_;

    /** amount of used absolute error for this query point */
    double used_error_;

    /** number of reference points taken care for this query point */
    index_t n_pruned_;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density_);
      OT_MY_OBJECT(used_error_);
      OT_MY_OBJECT(n_pruned_);
    }

  public:
    void Init(const Param& param) {      
      density_.Init(0, 0);
      used_error_ = 0;
      n_pruned_ = 0;
    }

    void Seed(const Param& param, const QPoint& q) {
    }

    /** divide each density by the normalization constant */
    void Postprocess(const Param& param, const QPoint& q, index_t q_index,
		     const RNode& r_root) {
      density_ *= param.mul_constant_;
    }

    /** apply left over postponed contributions */
    void ApplyPostponed(const Param& param, const QPostponed& postponed,
			const QPoint& q, index_t q_index) {
      density_ += postponed.d_density_;
      used_error_ += postponed.used_error_;
      n_pruned_ += postponed.n_pruned_;
    }
  };

  class QSummaryResult {
  public:

    /** bound on the density from leaves */
    DRange density_;

    /** maximum amount of error incurred among the query points */
    double used_error_;
    
    /** 
     * minimum bound on the portion of the reference dataset that has been
     * taken care of.
     */
    index_t n_pruned_;

    OT_DEF_BASIC(QSummaryResult) {
      OT_MY_OBJECT(density_);
      OT_MY_OBJECT(used_error_);
      OT_MY_OBJECT(n_pruned_);
    }

  public:
    
    /** initialize summary result to zeros */
    void Init(const Param& param) {
      density_.Init(0, 0);
      used_error_ = 0;
      n_pruned_ = 0;
    }

    void Seed(const Param& param, const QNode& q_node) {

    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      density_.InitEmptySet();
      used_error_ = 0;
      n_pruned_ = param.reference_count_;
    }

    /** 
     * refine query summary results by incorporating the given current
     * query result
     */
    void Accumulate(const Param& param, const QResult& result) {
      density_ |= result.density_;
      used_error_ = max(used_error_, result.used_error_);
      n_pruned_ = min(n_pruned_, result.n_pruned_);
    }

    /** 
     * this is the vertical operator that refines the current query summary
     * results based on the summary results owned by the given child
     */
    void Accumulate(const Param& param,
		    const QSummaryResult& result, index_t n_points) {
      density_ |= result.density_;
      used_error_ += result.used_error_;
      n_pruned_ = min(n_pruned_, result.n_pruned_);
    }

    void FinishReaccumulate(const Param& param, const QNode& q_node) {
    }

    /** 
     * horizontal join operator that accumulates the current best guess
     * on the density bound on the reference portion that has not been
     * visited so far.
     */
    void ApplySummaryResult(const Param& param,
			    const QSummaryResult& summary_result) {
      density_ += summary_result.density_;
      used_error_ += summary_result.used_error_;
      n_pruned_ += summary_result.n_pruned_;
    }

    /** apply deltas */
    void ApplyDelta(const Param& param, const Delta& delta) {
      density_ += delta.d_density_;
    }

    /** apply postponed contributions that were passed down */
    void ApplyPostponed(const Param& param,
			const QPostponed& postponed, const QNode& q_node) {
      density_ += postponed.d_density_;
      used_error_ += postponed.used_error_;
      n_pruned_ += postponed.n_pruned_;
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
  class PairVisitor {
  public:
    double density_;
    
  public:
    void Init(const Param& param) {}
    
    /** apply single-tree based pruning by iterating over each query point
     */
    bool StartVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node, const Delta& delta,
       const QSummaryResult& unapplied_summary_results, QResult* q_result,
       GlobalResult* global_result) {
      
      // compute distance bound between a given query point and the given
      // reference node and the resulting kernel value bound and density 
      // contribution bound
      DRange distance_sq_range = DRange(r_node.bound().MinDistanceSq(q.vec()),
					r_node.bound().MaxDistanceSq(q.vec()));
      DRange d_density = param.kernel_.RangeUnnormOnSq(distance_sq_range);
      d_density *= r_node.count();
      
      // refine the lower bound by incorporating the recently gained info and
      // unapplied info
      double density_lo = d_density.lo + q_result->density_.lo
	+ unapplied_summary_results.density_.lo;
      
      double allocated_error = 
	(param.relative_error_ * density_lo - q_result->used_error_) *
	(r_node.count() / 
	 ((double) param.reference_count_ - q_result->n_pruned_));
      
      q_result->n_pruned_ += r_node.count();
      
      // if we can prune the entire reference node for the given query point,
      // then we are done
      if (d_density.width() / 2.0 < allocated_error) {
        q_result->density_ += d_density;
	q_result->used_error_ += d_density.width() / 2.0;
        return false;
      }
      
      // otherwise, we need to iterate over each reference point
      density_ = 0;
      return true;
    }

    /** exhaustive computation between a query point and a reference point
     */
    void VisitPair(const Param& param, const QPoint& q, index_t q_index,
		   const RPoint& r, index_t r_index) {
      double distance_sq = la::DistanceSqEuclidean(q.vec(), r.vec());
      density_ += param.kernel_.EvalUnnormOnSq(distance_sq);
    }
    
    /** pass back the accumulated result into the query result
     */
    void FinishVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node, const QSummaryResult& unapplied_summary_results,
       QResult* q_result, GlobalResult* global_result) {
      
      q_result->density_ += density_;
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
				      const QNode& q_node,
				      const RNode& r_node,
				      const Delta& parent_delta,
				      Delta* delta,
				      GlobalResult* global_result,
				      QPostponed* q_postponed) {
      
      // compute distance bound between two nodes
      DRange distance_sq_range = 
	q_node.bound().RangeDistanceSq(r_node.bound());
      
      // compute the bound on kernel contribution
      delta->d_density_ = param.kernel_.RangeUnnormOnSq(distance_sq_range);
      delta->d_density_ *= r_node.count();
      
      DEBUG_ASSERT_MSG(delta->d_density_.lo <= 
		       delta->d_density_.hi * (1 + 1.0e-7),
		       "delta density lo %f > hi %f",
		       delta->d_density_.lo, delta->d_density_.hi);
      
      if (likely(delta->d_density_.hi != 0)) {
        return true;
      }
      // if the highest kernel value is zero, then perform exclusion pruning
      else {
        q_postponed->n_pruned_ += r_node.count();
	return false;
      }
    }

    /**
     * Prune based on the accumulated lower bound contribution and allocated
     * error
     */
    static bool ConsiderPairExtrinsic(const Param& param, const QNode& q_node,
				      const RNode& r_node, const Delta& delta,
				      const QSummaryResult& q_summary_result,
				      const GlobalResult& global_result,
				      QPostponed* q_postponed) {

      double allocated_error =
	(param.relative_error_ * q_summary_result.density_.lo -
	 q_summary_result.used_error_) *
	r_node.count() / (param.reference_count_ - q_summary_result.n_pruned_);

      if(delta.d_density_.width() / 2.0 <= allocated_error) {
	q_postponed->d_density_ += delta.d_density_;
	q_postponed->used_error_ += delta.d_density_.width() / 2.0;
	q_postponed->n_pruned_ += r_node.count();
	return false;
      }
      
      return true;
    }

    /**
     * Termination prune does not apply in KDE since all reference points
     * have to be considered...
     */
    static bool ConsiderQueryTermination
      (const Param& param, const QNode& q_node,
       const QSummaryResult& q_summary_result,
       const GlobalResult& global_result, QPostponed* q_postponed) {
      
      return true;
    }

    /**
     * Computes a heuristic for how early a computation should occur
     * -- smaller values are earlier.
     */
    static double Heuristic(const Param& param, const QNode& q_node,
			    const RNode& r_node, const Delta& delta) {
      return r_node.bound().MinToMidSq(q_node.bound());
    }
  };

  // functions

  /** compare naive and fast KDE results and compute maximum relative error */
  double ComputeMaximumRelativeError(datanode *module) {

    // create cache array for the distributed caches holding the fast KDE
    // naive KDE results
    CacheArray<QResult> q_fast_results_cache_array;
    q_fast_results_cache_array.Init(&q_results_, BlockDevice::M_READ);
 
    // maximum relative error
    double max_rel_err = 0;

    index_t q_end = (q_tree_->root()).end();

    for(index_t q = (q_tree_->root()).begin(); q < q_end; q++) {

      // create cache reader for each result
      CacheRead<QResult> q_fast_result(&q_fast_results_cache_array, q);

      double rel_err = fabs(0.5 * (q_fast_result->density_.lo +
				   q_fast_result->density_.hi) -
			    q_naive_results_[q]) / 
	fabs(q_naive_results_[q]);
      
      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    fx_format_result(module, "maximum relative error", "%g", max_rel_err);
    return max_rel_err;
  }

  /** KDE computation */
  void NaiveCompute() {
    
    // create cache array for the distriuted caches storing the query
    // reference points and query results
    CacheArray<QPoint> q_points_cache_array;
    CacheArray<RPoint> r_points_cache_array;
    q_points_cache_array.Init(q_points_cache_, BlockDevice::M_READ);
    r_points_cache_array.Init(r_points_cache_, BlockDevice::M_READ);
    
    index_t q_end = (q_tree_->root()).end();
    index_t r_end = (r_tree_->root()).end();
    for(index_t q = (q_tree_->root()).begin(); q < q_end; q++) {
      
      CacheRead<QPoint> q_point(&q_points_cache_array, q);

      if(q_point->vec().length() == 0) {
	printf("Problem with %d!\n", q);
      }

      for(index_t r = (r_tree_->root()).begin(); r < r_end; r++) {

	CacheRead<RPoint> r_point(&r_points_cache_array, r);

	// compute pairwise and add contribution
	double distance_sq = la::DistanceSqEuclidean(q_point->vec(), 
						     r_point->vec());

	double kernel_value = parameters_.kernel_.EvalUnnormOnSq(distance_sq);

	q_naive_results_[q] += kernel_value;

      } // finish looping over each reference point
      
      // normalize the current query density estimate
      q_naive_results_[q] *= parameters_.mul_constant_;
      
    } // finish looping over each query point
  }

  /** KDE computation using THOR */
  void Compute(datanode *module) {
    
    printf("Starting dualtree KDE...\n");
    fx_timer_start(module, "dualtree kde");
    thor::RpcDualTree<ThorKde, DualTreeDepthFirst<ThorKde> >
      (fx_submodule(fx_root, "gnp", "gnp"), GNP_CHANNEL,
       parameters_, q_tree_, r_tree_, &q_results_, &global_result_);
    fx_timer_stop(module, "dualtree kde");
    printf("Dualtree KDE completed...\n");

    printf("Starting naive KDE...\n");
    fx_timer_start(module, "naive kde");
    NaiveCompute();
    fx_timer_start(module, "naive kde");
    printf("Finished naive KDE...\n");
    ComputeMaximumRelativeError(module);
    
    rpc::Done();
  }

  /** read datasets, build trees */
  void Init(datanode *module) {

    // I don't quite understand what these mean, since I copied and pasted
    // from an example code.
    double results_megs = fx_param_double(module, "results/megs", 1000);

    rpc::Init();
    
    if (!rpc::is_root()) {
      fx_silence();
    }

    // initialize parameter set
    fx_submodule(module, NULL, "io");
    parameters_.Init(module);
       
    // read reference dataset
    fx_timer_start(module, "read_datasets");
    r_points_cache_ = new DistributedCache();
    parameters_.reference_count_ = 
      thor::ReadPoints<RPoint>(parameters_, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
			       fx_submodule(module, "data", "data"),
			       r_points_cache_);

    // read the query dataset if present
    if(fx_param_exists(module, "query")) {
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
    ThorKdePoint default_point;
    CacheArray<ThorKdePoint>::GetDefaultElement(r_points_cache_, 
						&default_point);   
    parameters_.FinalizeInit(module, default_point.vec().length());

    // construct trees
    fx_timer_start(module, "tree_construction");
    r_tree_ = new ThorTree<Param, RPoint, RNode>();
    thor::CreateKdTree<RPoint, RNode>(parameters_, DATA_CHANNEL + 4, 
				      DATA_CHANNEL + 5,
				      fx_submodule(module, "r_tree", "r_tree"),
				      parameters_.reference_count_, 
				      r_points_cache_, r_tree_);
    if (fx_param_exists(module, "query")) {
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

    // allocate space for naive computation
    q_naive_results_.Init(parameters_.query_count_);
    q_naive_results_.SetZero();
  }

  /** storing query results computed naively */
  Vector q_naive_results_;

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

};

#endif
