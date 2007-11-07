
/* this is garry's code with minor changes I have made whimsically - nishant */
// need to edit to make it handle the bichromatic case

#include "fastlib/fastlib_int.h"
#include "thor/thor.h"

//#define SOLVER_TYPE DualTreeRecursiveBreadth
#define SOLVER_TYPE DualTreeDepthFirst


//#define LOO 1 //for LOO, make this 1, else make this 0

int LOO = 1;


/**
 * Approximate kernel density estimation.
 *
 * Uses Epanechnikov kernels and the inclusion rule.
 */
template<typename TKernel>
class FdKde {
public:
  /** The bounding type. Required by THOR. */
  typedef DHrectBound<2> Bound;
  
  /** Point data includes index and vector */
  class Point {
  private:
    index_t index_;
    Vector vec_;
    
    OT_DEF_BASIC(Point) {
      OT_MY_OBJECT(index_);
      OT_MY_OBJECT(vec_);
    }
    
  public:

    /**
     * Gets the index.
     */
    index_t index() const {
      return index_;
    }

    /**
     * Gets the vector.
     */
    const Vector& vec() const {
      return vec_;
    }

    /**
     * Gets the vector.
     */
    Vector& vec() {
      return vec_;
    }
    
    /**
     * Initializes a "default element" from a dataset schema.
     *
     * This is the only function that allows allocation.
     */
    template<typename Param>
    void Init(const Param& param, const DatasetInfo& schema) {
      vec_.Init(schema.n_features());
      vec_.SetZero();
    }
    /**
     * Sets the values of this object, not allocating any memory.
     *
     * If memory needs to be allocated it must be allocated at the beginning
     * with Init.
     *
     * @param param ignored
     * @param index the index of the point, ignored
     * @param data the vector data read from file
     */
    template<typename Param>
    void Set(const Param& param, index_t index, const Vector& data) {
      index_ = index;
      vec_.CopyValues(data);
    }
  };
    
  typedef Point QPoint;
  typedef Point RPoint;

  /** The type of kernel in use.  NOT required by THOR. */
  typedef TKernel Kernel;

  /**
   * All parameters required by the execution of the algorithm.
   *
   * Required by N-Body Reduce.
   */
  struct Param {
  public:
    /** The kernel in use. */
    Kernel kernel;
    // /** The amount of relative error prooportional to local lower bound. */
    // double rel_error_local;

    /** The dimensionality of the data sets. */
    index_t dim;
    /** Number of query points. */
    index_t q_count;
    /** Number of reference points. */
    index_t r_count;
    /** The multiplicative norm constant for the kernel. */
    double mul_constant;
    /** The amount of relative error allowed. */
    double rel_error;
    // /** Amount of error that is local error */
    // double p_local;
    // /** Amount of error that is global error */
    // double p_global;
    /** The band width, h. */
    double bandwidth;

    OT_DEF_BASIC(Param) {
      OT_MY_OBJECT(kernel);
      //OT_MY_OBJECT(rel_error_local);
      OT_MY_OBJECT(dim);
      OT_MY_OBJECT(q_count);
      OT_MY_OBJECT(r_count);
      OT_MY_OBJECT(mul_constant);
      OT_MY_OBJECT(rel_error);
      //OT_MY_OBJECT(p_local);
      //OT_MY_OBJECT(p_global);
      OT_MY_OBJECT(bandwidth);
    }

  public:
    /**
     * Initialize parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      bandwidth = fx_param_double_req(module, "h");
      //p_local = fx_param_double(module, "p_local", 0);
      //p_global = 1 - p_local;
      rel_error = fx_param_double(module, "rel_error", 0.1);
      printf("\n\n\nrel_error = %f\n\n\n", rel_error);
      //rel_error_local = rel_error * p_local;
    }

    /** this is called after things are set. */
    void SetDimensions() {
      kernel.Init(bandwidth, dim);
      mul_constant = 1.0 / (kernel.CalcNormConstant(dim) * (r_count - LOO));
    }
  };

  /**
   * Per-reference-node bottom-up statistic.
   *
   * The statistic must be commutative and associative, thus bottom-up
   * computable.
   */
  struct RStat {
  public:
    //MomentInfo moment_info;

    OT_DEF_BASIC(RStat) {
      //OT_MY_OBJECT(moment_info);
    }

  public:
    /**
     * Initialize to a default zero value, as if no data is seen (Req THOR).
     *
     * This is the only method in which memory allocation can occur.
     */
    void Init(const Param& param) {
      //moment_info.Init(param);
    }

    /**
     * Accumulate data from a single point (Req THOR).
     */
    void Accumulate(const Param& param, const QPoint& point) {
      //moment_info.Add(1, point.vec(), la::Dot(point.vec(), point.vec()));
    }

    /**
     * Accumulate data from one of your children (Req THOR).
     */
    void Accumulate(const Param& param,
		    const RStat& stat, const Bound& bound, index_t n) {
      //moment_info.Add(stat.moment_info);
    }

    /**
     * Finish accumulating data; for instance, for mean, divide by the
     * number of points.
     */
    void Postprocess(const Param& param, const Bound& bound, index_t n) {
    }
  };

  /**
   * Query-tree statistic.
   *
   * Note that this statistic is not actually needed and a blank statistic
   * is fine, but QStat must equal RStat in order for us to allow
   * monochromatic execution.
   *
   * This limitation may be removed in a further version of THOR.
   */
  typedef RStat QStat;

  typedef ThorNode<Bound, RStat> Node;
  typedef Node QNode;
  typedef Node RNode;

  /**
   * Coarse result on a region.
   */
  struct QPostponed {
  public:
    /** The density contribution postponed. */
    DRange d_density;
    index_t n_pruned;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(d_density);
      OT_MY_OBJECT(n_pruned);
    }

  public:
    void Init(const Param& param) {
      d_density.Init(0, 0);
      n_pruned = 0;
    }

    void Reset(const Param& param) {
      d_density.Init(0, 0);
      n_pruned = 0;
    }

    void ApplyPostponed(const Param& param, const QPostponed& other) {
      d_density += other.d_density;
      n_pruned += other.n_pruned;
    }
  };

  /**
   * Coarse result on a region.
   */
  struct Delta {
  public:
    /** Min and max density update possible. */
    DRange d_density;

    OT_DEF_BASIC(Delta) {
      OT_MY_OBJECT(d_density);
    }

  public:
    void Init(const Param& param) {
    }
  };

  // rho
  struct QResult {
  public:
    DRange density;
    index_t n_pruned;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(n_pruned);
    }

  public:
    void Init(const Param& param) {
      density.Init(0, 0);
      n_pruned = 0;
    }

    void Postprocess(const Param& param,
		     const QPoint& q, index_t q_index,
		     const RNode& r_root) {
      density -= LOO; // LOO
      density.lo *= param.mul_constant;
      density.hi *= param.mul_constant;
    }

    void ApplyPostponed(const Param& param,
			const QPostponed& postponed,
			const QPoint& q, index_t q_index) {
      density += postponed.d_density;
      n_pruned += postponed.n_pruned;
    }
  };

  struct QSummaryResult {
  public:
    /** Bound on density from leaves. */
    DRange density;
    double used_width;
    index_t n_pruned;

    OT_DEF_BASIC(QSummaryResult) {
      OT_MY_OBJECT(density);
      OT_MY_OBJECT(used_width);
      OT_MY_OBJECT(n_pruned);
    }

  public:
    void Init(const Param& param) {
      /* horizontal init */
      density.Init(0, 0);
      used_width = 0;
      n_pruned = 0;
    }

    /** horizontal join operator */
    void ApplySummaryResult(const Param& param,
			    const QSummaryResult& summary_result) {
      density += summary_result.density;
      used_width += summary_result.used_width;
      n_pruned += summary_result.n_pruned;
    }

    void ApplyDelta(const Param& param,
		    const Delta& delta) {
      density += delta.d_density;
      // deltas don't affect used error
    }

    void ApplyPostponed(const Param& param,
			const QPostponed& postponed, const QNode& q_node) {
      density += postponed.d_density;
      used_width += postponed.d_density.width();
      n_pruned += postponed.n_pruned;
    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
      /* vertical init */
      density.InitEmptySet();
      used_width = 0;
      n_pruned = param.r_count;
    }

    void Accumulate(const Param& param, const QResult& result) {
      // TODO: applying to single result could be made part of QResult,
      // but in some cases may require a copy/undo stage
      density |= result.density;
      used_width = max(used_width, result.density.width());
      n_pruned = min(n_pruned, result.n_pruned);
    }

    void Accumulate(const Param& param,
		    const QSummaryResult& result, index_t n_points) {
      density |= result.density;
      used_width = max(used_width, result.used_width);
      n_pruned = min(n_pruned, result.n_pruned);
    }

    void FinishReaccumulate(const Param& param,
			    const QNode& q_node) {
      /* no post-processing steps necessary */
    }
  };

  /**
   * A simple postprocess-step global result.
   */
  struct GlobalResult {
  public:
    DRange sum_density;
    double foo;
    
    OT_DEF(GlobalResult) {
      OT_MY_OBJECT(sum_density);
      OT_MY_OBJECT(foo);
    }

  public:
    void Init(const Param& param) {
      sum_density.Init(0, 0);
      foo = 0;
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
      sum_density += other.sum_density;
      foo += other.foo;
    }
    void Postprocess(const Param& param) {}
    void Report(const Param& param, datanode *datanode) {
      fx_format_result(datanode, "avg_density_lo", "%g", sum_density.lo / param.q_count);
      fx_format_result(datanode, "avg_density_hi", "%g", sum_density.hi / param.q_count);
      fx_format_result(datanode, "avg_density", "%g", sum_density.mid() / param.q_count);
      fx_format_result(datanode, "avg_rel_error", "%g",
		       sum_density.width() / sum_density.lo / 2);
      fx_format_result(datanode, "foo", "%g", foo / param.q_count);
    }
    void ApplyResult(const Param& param,
		     const QPoint& q_point, index_t q_i,
		     const QResult& result) {
      sum_density += result.density;
      foo += log(result.density.mid());
    }
  };

  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  struct PairVisitor {
  public:
    double density;

  public:
    void Init(const Param& param) {}

    // notes
    // - this function must assume that global_result is incomplete (which is
    // reasonable in allnn)
    bool StartVisitingQueryPoint(const Param& param,
				 const QPoint& q, index_t q_index,
				 const RNode& r_node,
				 const QSummaryResult& unapplied_summary_results,
				 QResult* q_result,
				 GlobalResult* global_result) {
      DRange distance_sq_range = DRange(
					r_node.bound().MinDistanceSq(q.vec()),
					r_node.bound().MaxDistanceSq(q.vec()));
      DRange d_density = param.kernel.RangeUnnormOnSq(distance_sq_range);

      d_density *= r_node.count();

      double density_lo = d_density.lo + q_result->density.lo
	+ unapplied_summary_results.density.lo;

      double allocated_width =
	(param.rel_error * density_lo * 2 - q_result->density.width())
	* r_node.count() / (param.r_count - q_result->n_pruned);
      /*allocated_error *= param.p_global;
	allocated_error += param.rel_error_local * d_density.lo;*/

      q_result->n_pruned += r_node.count();

      if (d_density.width() < allocated_width) {
	q_result->density += d_density;
	return false;
      }

      density = 0;

      return true;
    }

    /**
     * This is the lame form of the function used by breadth-first.
     *
     * Since breadth-first tries to avoid getting to leaves anyways, it
     * doesn't want to bother with giving you summary results, so it doesn't.
     */
    bool StartVisitingQueryPoint(const Param& param,
				 const QPoint& q, index_t q_index,
				 const RNode& r_node,
				 QResult* q_result,
				 GlobalResult* global_result) {
      density = 0;
      return true;
    }

    void VisitPair(const Param& param,
		   const QPoint& q, index_t q_index,
		   const RPoint& r, index_t r_index) {
      double distance = la::DistanceSqEuclidean(q.vec(), r.vec());
      density += param.kernel.EvalUnnormOnSq(distance);
    }

    void FinishVisitingQueryPoint(const Param& param,
				  const QPoint& q, index_t q_index, const RNode& r_node,
				  const QSummaryResult& unapplied_summary_results,
				  QResult* q_result, GlobalResult* global_result) {
      q_result->density += density;
    }

    /**
     * Once again, the lame form for breadth-first.
     */
    void FinishVisitingQueryPoint(const Param& param,
				  const QPoint& q, index_t q_index, const RNode& r_node,
				  QResult* q_result, GlobalResult* global_result) {
      q_result->density += density;
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
      //DRange distance_sq =
      //    q_node.bound().RangeDistanceSq(r_node.bound());
      //double distance_sq_mid = q_node.bound().MaxToMidSq(r_node.bound());
      DRange distance_sq_range = q_node.bound().RangeDistanceSq(r_node.bound());

      delta->d_density = param.kernel.RangeUnnormOnSq(distance_sq_range);
      delta->d_density *= r_node.count();

      DEBUG_ASSERT_MSG(delta->d_density.lo <= delta->d_density.hi * (1 + 1.0e-7),
		       "delta density lo %f > hi %f",
		       delta->d_density.lo, delta->d_density.hi);


      if (likely(delta->d_density.hi != 0)) {
	return true;
      } else {
	q_postponed->n_pruned += r_node.count();
	return false;
      }
    }

    static bool ConsiderPairExtrinsic(
				      const Param& param,
				      const QNode& q_node,
				      const RNode& r_node,
				      const Delta& delta,
				      const QSummaryResult& q_summary_result,
				      const GlobalResult& global_result,
				      QPostponed* q_postponed) {
      double allocated_width =
	(param.rel_error * q_summary_result.density.lo * 2
	 - q_summary_result.used_width)
	* r_node.count() / (param.r_count - q_summary_result.n_pruned);
      /*allocated_width *= param.p_global;
	allocated_width += param.rel_error_local * delta.d_density.lo * 2;*/
      //fprintf(stderr, "%e..%e (%e, %e) %e (%e)\n", delta.d_density.lo, delta.d_density.hi, delta.d_density.width(), allocated_width, q_summary_result.density.lo,
      //    sqrt(q_node.bound().MaxDistanceSq(q_node.bound())));

      if (delta.d_density.width() <= allocated_width) {
	q_postponed->d_density += delta.d_density;
	q_postponed->n_pruned += r_node.count();
	return false;
      }

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
     * Computes a heuristic for how early a computation should occur -- smaller
     * values are earlier.
     */
    static double Heuristic(
			    const Param& param,
			    const QNode& q_node,
			    const RNode& r_node,
			    const Delta& delta) {
      //double d = r_node.bound().MinToMidSq(q_node.bound());
      //return delta.d_density.lo / delta.d_density.width() + d * 1.0e-32;
      double d = r_node.bound().MidDistanceSq(q_node.bound());
      return d;
      //return fabs(d - param.kernel.bandwidth_sq());
    }
  };

  static double DoKdeEntropyL2E(datanode *module,
				Vector h_KL_log_densities,
				Vector &densities) {
    
    if (!rpc::is_root()) {
      // turn off fastexec output
      fx_silence();
    }

    const int TREE_CHANNEL = 300;
    const int RESULT_CHANNEL = 350;



    Param *param;
    ThorTree<Param, Point, Node> q_tree;
    ThorTree<Param, Point, Node> r_tree;
    
    param = new Param();
    param->Init(fx_submodule(module, "kde", "kde"));


    DistributedCache results;
    double results_megs = fx_param_double(module, "results/megs", 1000);
    
    
    
    fx_timer_start(module, "read");
    DistributedCache *q_points_cache = new DistributedCache();
    param->q_count = thor::ReadPoints<Point>(
					     Empty(), TREE_CHANNEL + 0, TREE_CHANNEL + 1,
					     fx_submodule(module, "data_linspace", "data_linspace"), q_points_cache);
    fx_timer_stop(module, "read");
    
    fx_timer_start(module, "read");
    DistributedCache *r_points_cache = new DistributedCache();
    param->r_count = thor::ReadPoints<Point>(
					     Empty(), TREE_CHANNEL + 0, TREE_CHANNEL + 1,
					     fx_submodule(module, "data", "data"), r_points_cache);
    fx_timer_stop(module, "read");
    
    
    
    Point example_point;
    CacheArray<Point>::GetDefaultElement(r_points_cache,
					 &example_point);
    param->dim = example_point.vec().length();
    
    
    LOO = 0;
    param->SetDimensions();
    
    
    
    //edit for query
    fx_timer_start(module, "q_tree");
    thor::CreateKdTree<Point, Node>(
				    *param, TREE_CHANNEL + 2, TREE_CHANNEL + 3,
				    fx_submodule(module, "q_tree", "q_tree"), param->q_count,
				    q_points_cache, &q_tree);
    fx_timer_stop(module, "q_tree");
    
    
    fx_timer_start(module, "r_tree");
    thor::CreateKdTree<Point, Node>(
				    *param, TREE_CHANNEL + 2, TREE_CHANNEL + 3,
				    fx_submodule(module, "r_tree", "r_tree"), param->r_count,
				    r_points_cache, &r_tree);
    fx_timer_stop(module, "r_tree");
    
    QResult example_result;
    example_result.Init(*param);
    //changed from tree to q_tree
    q_tree.CreateResultCache(RESULT_CHANNEL, example_result,
			     results_megs, &results);
    
    GlobalResult global_result_1;
    
    densities.Init(param->q_count);
    




    ArrayList<double> test_bandwidths;
    test_bandwidths.Init();

    ArrayList<double> test_scores;
    test_scores.Init();
    


    index_t test_state = 0;

    double left_bandwidth = 1e-5;
    double right_bandwidth = 1;
    double mid_bandwidth = -1;

    double left_score = -1;
    double right_score = -1;
    double mid_score = -1;

    const double EPSILON = 1e-5;

    bool search = true;

    /*
      1) Test left_bandwidth and right_bandwidth
      2) iteratively test (left_bandwidth + right_bandwidth) / 2
      until (eval(left_bandwidth) - eval(right_bandwidth)) < epsilon
    */

    
    
    for(index_t iter = 0; search; iter++) {


      bool choose = false;

      
      if (rpc::is_root()) {
	fprintf(stderr, "Doing density estimation now...\n");
	if(LOO) {
	  printf("LOO enabled\n");
	}
	else{
	  printf("LOO disabled\n");
	}
      }

      printf("left = %f\n", left_bandwidth);
      printf("mid = %f\n", mid_bandwidth);
      printf("right = %f\n", right_bandwidth);

      
      switch(test_state) {
      case 0:
	param->bandwidth = left_bandwidth;
	break;
      case 1:
	param->bandwidth = right_bandwidth;
	break;
      case 2:
	if((right_bandwidth - left_bandwidth) > EPSILON) {
	  param->bandwidth = (left_bandwidth + right_bandwidth) / 2;
	}
	else {
	  search = false;
	}
	break;
      case 3:

	double right_diff = right_bandwidth - mid_bandwidth;
	double left_diff = mid_bandwidth - left_bandwidth;

	if((right_diff < EPSILON) && (left_diff < EPSILON)) {
	  search = false;
	}
	else {
	  if(right_diff > left_diff) {
	    choose = 1; //choose right
	  }
	  else if(left_diff > right_diff) {
	    choose = 0; // choose left
	  }
	  else {
	    if(left_score < right_score) {
	      choose = 0; // choose left
	    }
	    else {
	      choose = 1; // choose right
	    }
	  }

	  if(choose == 0) {
	    param->bandwidth = (left_bandwidth + mid_bandwidth) / 2;
	    printf("left = %f, mid = %f\n", left_bandwidth, mid_bandwidth);
	    printf("splitting on left and mid: ");
	  }
	  else {
	    param->bandwidth = (right_bandwidth + mid_bandwidth) / 2;
	    printf("right = %f, mid = %f\n", right_bandwidth, mid_bandwidth);
	    printf("splitting on right and mid: ");
	  }
	  printf("%f\n", param->bandwidth);
	}
	break;
      default:
	;//impossible!
      }


      if(!search) {
	break;
      }

      //search = false; //early termination for debugging
            
      param->SetDimensions();
      
      printf("\nusing bandwidth %f\n", param->bandwidth);
    

      
      fx_timer_start(module, "kde_x");
      thor::RpcDualTree<FdKde, SOLVER_TYPE<FdKde> >(
						    fx_submodule(module, "gnp", "kde_%d", iter), 200,
						    *param, &q_tree, &r_tree, &results, &global_result_1);
      fx_timer_stop(module, "kde_x");
      
      
      
      // emit results
      
      
      
      if (rpc::is_root()) {
	CacheArray<QResult> result_array;
	CacheArray<QPoint> q_points_array;
	result_array.Init(&results, BlockDevice::M_READ);
	q_points_array.Init(q_points_cache, BlockDevice::M_READ);
	CacheReadIter<QResult> result_iter(&result_array, 0);
	CacheReadIter<QPoint> q_points_iter(&q_points_array, 0);
	for (index_t i = 0; i < param->q_count; i++,
	       result_iter.Next(), q_points_iter.Next()) {
	  //I think lo and hi should always be the same so we just use lo
	  densities[(*q_points_iter).index()] = (*result_iter).density.lo;
	}



	//sum over the first r_count queries
	double new_mul_constant = 1 / (param->kernel.CalcNormConstant(param->dim) * (param->r_count - 1));
	for(index_t i = 0; i < param->r_count; i++) {
	  double p = densities[i];
	  //since density was calculated with LOO disabled, adjust density
	  //mul_constant = 1.0 / (kernel.CalcNormConstant(dim) * r_count - LOO);
	  p = ((p / param->mul_constant) - 1) * new_mul_constant;
	  densities[i] = p * log(p);
	}

	double first_term =
	  la::Dot(h_KL_log_densities, densities) /
	  ((double)(param->r_count));


	// only sum over the last q_count - r_count queries
	double sum = 0;
	for(index_t i = param->r_count; i < param->q_count; i++) {
	  double p = densities[i];
	  sum += pow(p * log(p), 2);
	}

	double second_term = sum / ((double)(param->q_count));

	double score = first_term + second_term;

	printf("score = %f\n", score);

    
	test_bandwidths.AddBackItem(param->bandwidth);
	test_scores.AddBackItem(score);
	    

	switch(test_state) {
	case 0:
	  left_score = score;
	  test_state = 1;
	  break;
	case 1:
	  right_score = score;
	  test_state = 2;
	  break;
	case 2:
	  if((right_score < score) &&
	     (score < left_score)) {
	    left_bandwidth = param->bandwidth;
	    left_score = score;
	  }
	  else if((left_score < score) &&
		  (score < right_score)) {
	    right_bandwidth = param->bandwidth;
	    right_score = score;
	  }
	  else {
	    mid_bandwidth = param->bandwidth;
	    mid_score = score;
	    test_state = 3;
	  }
	  break;
	case 3:

	  double x_score;

	  if(choose == 0) {
	    x_score = left_score;
	  }
	  else {
	    x_score = right_score;
	  }

	  if((mid_score <= score) && (score <= x_score)) {
	    if(choose == 0) {
	      left_bandwidth = param->bandwidth;
	      left_score = score;
	    }
	    else {
	      right_bandwidth = param->bandwidth;
	      right_score = score;
	    }
	  }
	  else {
	    if(choose == 0) {
	      right_bandwidth = mid_bandwidth;
	      right_score = mid_score;
	      mid_bandwidth = param->bandwidth;
	      mid_score = score;
	    }
	    else {
	      left_bandwidth = mid_bandwidth;
	      left_score = mid_score;
	      mid_bandwidth = param->bandwidth;
	      mid_score = score;
	    }
	  }
	  
	  break;
	default:
	  ;//impossible!
	}
	
      }
      //data::Save(fx_param_str(module, "results", "results.csv"), densities);
      results.ResetElements();

    }
    
    
    printf("bandwidth\tscore\n");
    
    for(index_t i = 0; i < test_bandwidths.size(); i++) {
      printf("%f\t%f\n", test_bandwidths[i], test_scores[i]);

    }
    

    
    delete param;

    return 1;



    
  }


  static double DoKdeKL(datanode *module, Vector &log_densities) {

    if (!rpc::is_root()) {
      // turn off fastexec output
      fx_silence();
    }
    
    const int TREE_CHANNEL = 300;
    const int RESULT_CHANNEL = 350;
    
    
    
    Param *param;
    ThorTree<Param, Point, Node> q_tree;
    ThorTree<Param, Point, Node> r_tree;
    
    param = new Param();
    param->Init(fx_submodule(module, "kde", "kde"));
    
    DistributedCache results;
    double results_megs = fx_param_double(module, "results/megs", 1000);
      
      

    fx_timer_start(module, "read");
    DistributedCache *q_points_cache = new DistributedCache();
    param->q_count = thor::ReadPoints<Point>(
					     Empty(), TREE_CHANNEL + 0, TREE_CHANNEL + 1,
					     fx_submodule(module, "data", "data"), q_points_cache);
    fx_timer_stop(module, "read");

    fx_timer_start(module, "read");
    DistributedCache *r_points_cache = new DistributedCache();
    param->r_count = thor::ReadPoints<Point>(
					     Empty(), TREE_CHANNEL + 0, TREE_CHANNEL + 1,
					     fx_submodule(module, "data", "data"), r_points_cache);
    fx_timer_stop(module, "read");


      
    Point example_point;
    CacheArray<Point>::GetDefaultElement(r_points_cache,
					 &example_point);
    param->dim = example_point.vec().length();

    
    LOO = 1;
    param->SetDimensions();

    

    //edit for query
    fx_timer_start(module, "q_tree");
    thor::CreateKdTree<Point, Node>(
				    *param, TREE_CHANNEL + 2, TREE_CHANNEL + 3,
				    fx_submodule(module, "q_tree", "q_tree"), param->q_count,
				    q_points_cache, &q_tree);
    fx_timer_stop(module, "q_tree");

      
    fx_timer_start(module, "r_tree");
    thor::CreateKdTree<Point, Node>(
				    *param, TREE_CHANNEL + 2, TREE_CHANNEL + 3,
				    fx_submodule(module, "r_tree", "r_tree"), param->r_count,
				    r_points_cache, &r_tree);
    fx_timer_stop(module, "r_tree");
    
    QResult example_result;
    example_result.Init(*param);
    //changed from tree to q_tree
    q_tree.CreateResultCache(RESULT_CHANNEL, example_result,
			     results_megs, &results);
    
    GlobalResult global_result_1;
    
    log_densities.Init(param->q_count);
    


    ArrayList<double> test_bandwidths;
    test_bandwidths.Init();

    ArrayList<double> test_scores;
    test_scores.Init();
    


    index_t test_state = 0;

    double left_bandwidth = 1e-5;
    double right_bandwidth = 1;
    double mid_bandwidth = -1;

    double left_score = -1;
    double right_score = -1;
    double mid_score = -1;

    const double EPSILON = 1e-5;

    bool search = true;

    /*
      1) Test left_bandwidth and right_bandwidth
      2) iteratively test (left_bandwidth + right_bandwidth) / 2
      until (eval(left_bandwidth) - eval(right_bandwidth)) < epsilon */

    
    
    for(index_t iter = 0; search; iter++) {


      bool choose = false;

      
      if (rpc::is_root()) {
	fprintf(stderr, "Doing density estimation now...\n");
	if(LOO) {
	  printf("LOO enabled\n");
	}
	else{
	  printf("LOO disabled\n");
	}
      }

      printf("left = %f\n", left_bandwidth);
      printf("mid = %f\n", mid_bandwidth);
      printf("right = %f\n", right_bandwidth);

      
      switch(test_state) {
      case 0:
	param->bandwidth = left_bandwidth;
	break;
      case 1:
	param->bandwidth = right_bandwidth;
	break;
      case 2:
	if((right_bandwidth - left_bandwidth) > EPSILON) {
	  param->bandwidth = (left_bandwidth + right_bandwidth) / 2;
	}
	else {
	  search = false;
	}
	break;
      case 3:

	double right_diff = right_bandwidth - mid_bandwidth;
	double left_diff = mid_bandwidth - left_bandwidth;

	if((right_diff < EPSILON) && (left_diff < EPSILON)) {
	  search = false;
	}
	else {
	  if(right_diff > left_diff) {
	    choose = 1; //choose right
	  }
	  else if(left_diff > right_diff) {
	    choose = 0; // choose left
	  }
	  else {
	    if(left_score < right_score) {
	      choose = 0; // choose left
	    }
	    else {
	      choose = 1; // choose right
	    }
	  }

	  if(choose == 0) {
	    param->bandwidth = (left_bandwidth + mid_bandwidth) / 2;
	    printf("left = %f, mid = %f\n", left_bandwidth, mid_bandwidth);
	    printf("splitting on left and mid: ");
	  }
	  else {
	    param->bandwidth = (right_bandwidth + mid_bandwidth) / 2;
	    printf("right = %f, mid = %f\n", right_bandwidth, mid_bandwidth);
	    printf("splitting on right and mid: ");
	  }
	  printf("%f\n", param->bandwidth);
	}
	break;
      default:
	;//impossible!
      }


      if(!search) {
	break;
      }

      //search = false; //early termination for debugging

      
      param->SetDimensions();
      
      printf("\nusing bandwidth %f\n", param->bandwidth);
    

      
      fx_timer_start(module, "kde_x");
      thor::RpcDualTree<FdKde, SOLVER_TYPE<FdKde> >(
						    fx_submodule(module, "gnp", "kde_%d", iter), 200,
						    *param, &q_tree, &r_tree, &results, &global_result_1);
      fx_timer_stop(module, "kde_x");
      
      
      
      // emit results
      
      
      
      if (rpc::is_root()) {
	CacheArray<QResult> result_array;
	CacheArray<QPoint> q_points_array;
	result_array.Init(&results, BlockDevice::M_READ);
	q_points_array.Init(q_points_cache, BlockDevice::M_READ);
	CacheReadIter<QResult> result_iter(&result_array, 0);
	CacheReadIter<QPoint> q_points_iter(&q_points_array, 0);
	for (index_t i = 0; i < param->q_count; i++,
	       result_iter.Next(), q_points_iter.Next()) {
	  //I think lo and hi should always be the same so we just use lo
	  log_densities[(*q_points_iter).index()] = 
	    log((*result_iter).density.lo);
	}


	double sum_log_density = 0;
	for(index_t i = 0; i < param->q_count; i++) {
	  sum_log_density += log_densities[i];
	}

	double expected_log_density = -sum_log_density / ((double)param->q_count);

	printf("\nexpected log density = %f\n", expected_log_density);





	test_bandwidths.AddBackItem(param->bandwidth);
	test_scores.AddBackItem(expected_log_density);
	    

	switch(test_state) {
	case 0:
	  left_score = expected_log_density;
	  test_state = 1;
	  break;
	case 1:
	  right_score = expected_log_density;
	  test_state = 2;
	  break;
	case 2:
	  if((right_score < expected_log_density) &&
	     (expected_log_density < left_score)) {
	    left_bandwidth = param->bandwidth;
	    left_score = expected_log_density;
	  }
	  else if((left_score < expected_log_density) &&
		  (expected_log_density < right_score)) {
	    right_bandwidth = param->bandwidth;
	    right_score = expected_log_density;
	  }
	  else {
	    mid_bandwidth = param->bandwidth;
	    mid_score = expected_log_density;
	    test_state = 3;
	  }
	  break;
	case 3:

	  double x_score;

	  if(choose == 0) {
	    x_score = left_score;
	  }
	  else {
	    x_score = right_score;
	  }

	  if((mid_score <= expected_log_density) && (expected_log_density <= x_score)) {
	    if(choose == 0) {
	      left_bandwidth = param->bandwidth;
	      left_score = expected_log_density;
	    }
	    else {
	      right_bandwidth = param->bandwidth;
	      right_score = expected_log_density;
	    }
	  }
	  else {
	    if(choose == 0) {
	      right_bandwidth = mid_bandwidth;
	      right_score = mid_score;
	      mid_bandwidth = param->bandwidth;
	      mid_score = expected_log_density;
	    }
	    else {
	      left_bandwidth = mid_bandwidth;
	      left_score = mid_score;
	      mid_bandwidth = param->bandwidth;
	      mid_score = expected_log_density;
	    }
	  }
	  
	  break;
	default:
	  ;//impossible!
	}
	
      }
      //data::Save(fx_param_str(module, "results", "results.csv"), log_densities);
      results.ResetElements();

    }
    
    
    printf("bandwidth\tscore\n");

    //    double best_score = 0;
    
    for(index_t i = 0; i < test_bandwidths.size(); i++) {
      printf("%f\t%f\n", test_bandwidths[i], test_scores[i]);

    }


    // NOTE: I need to ensure that the last bandwidth tested is in fact optimal or close enough
    return *(test_bandwidths.last());
  }





};

void KdeMain(datanode *module) {
  String kernel = fx_param_str(module, "kde/kernel", "gauss");


  printf("Kullback-Leibler Loss\n");
  
  Vector h_KL_log_densities;
  double h_KL = -1;
  
  // Kullback-Leibler Loss
  if (kernel == "gauss" || kernel == "gauss_star") {
    h_KL = FdKde<GaussianKernel>::DoKdeKL(module, h_KL_log_densities);
  }
  else if (kernel == "epan") {
    h_KL = FdKde<EpanKernel>::DoKdeKL(module, h_KL_log_densities);
  }
  else {
    FATAL("Unsupported kernel: '%s'", kernel.c_str());
  }



  printf("Entropy L2E\n");

  Vector densities;
  double h = -1;

  if (kernel == "gauss" || kernel == "gauss_star") {
    h = FdKde<GaussianKernel>::DoKdeEntropyL2E(module,
					       h_KL_log_densities,
					       densities);
  }
  else if (kernel == "epan") {
    h = FdKde<EpanKernel>::DoKdeEntropyL2E(module, 
					   h_KL_log_densities,
					   densities);
  }
  else {
    FATAL("Unsupported kernel: '%s'", kernel.c_str());
  }



  

  
  
  


}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  rpc::Init();
  
  KdeMain(fx_root);
  
  rpc::Done();
  //  fx_done();
  

  

}

  
