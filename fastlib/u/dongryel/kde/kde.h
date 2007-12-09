#ifndef KDE_H
#define KDE_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/mult_farfield_expansion.h"
#include "u/dongryel/series_expansion/mult_local_expansion.h"
#include "u/dongryel/series_expansion/kernel_aux.h"

template<typename TKernel>
class NaiveKde {
  
 private:
  
  /** query dataset */
  Matrix qset_;
  
  /** reference dataset */
  Matrix rset_;
  
  /** kernel */
  TKernel kernel_;

  /** computed densities */
  Vector densities_;
  
 public:
  
  void Compute() {

    printf("\nStarting naive KDE...\n");
    fx_timer_start(NULL, "naive_kde_compute");

    // compute unnormalized sum
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	const double *r_col = rset_.GetColumnPtr(r);
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	
	densities_[q] += kernel_.EvalUnnormOnSq(dsqd);
      }
    }
    
    // then normalize it
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows()) * 
      rset_.n_cols();
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
    fx_timer_stop(NULL, "naive_kde_compute");
    printf("\nNaive KDE completed...\n");
  }

  void Init() {
    densities_.SetZero();
  }

  void Init(Matrix &qset, Matrix &rset) {

    // get datasets
    qset_.Alias(qset);
    rset_.Alias(rset);

    // get bandwidth
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));
    
    // allocate density storage
    densities_.Init(qset.n_cols());
    densities_.SetZero();
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if(fx_param_exists(NULL, "naive_kde_output")) {
      fname = fx_param_str(NULL, "naive_kde_output", NULL);
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", densities_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }    
  }
  
  void ComputeMaximumRelativeError(const Vector &density_estimates) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < densities_.length(); q++) {
      double rel_err = fabs(density_estimates[q] - densities_[q]) / 
	densities_[q];

      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    fx_format_result(NULL, "maxium_relative_error_for_fast_KDE", "%g", 
		     max_rel_err);
  }

};


template<typename TKernelAux>
class FastKde {
  
 private:
  
  // forward declaration of KdeStat class
  class KdeStat;
  
  // our tree type using the KdeStat
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, KdeStat > Tree;

  /** parameter class */
  class Param {
  public:

    /** series expansion auxililary object */
    TKernelAux ka_;

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
      OT_MY_OBJECT(ka_);
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
      ka_.kernel_.Init(bandwidth_);
      mul_constant_ = 1.0 /
        (ka_.kernel_.CalcNormConstant(dimension) * reference_count_);

      // initialize the series expansion object
      if(fx_param_exists(module, "multiplicative_expansion")) {
        if(dimension_ <= 2) {
          ka_.sea_.Init(fx_param_int(module, "order", 5), dimension);
        }
        else if(dimension_ <= 3) {
          ka_.sea_.Init(fx_param_int(module, "order", 1), dimension);
        }
        else {
          ka_.sea_.Init(fx_param_int(module, "order", 0), dimension);
        }
      }
      else {
        if(dimension_ <= 2) {
          ka_.sea_.Init(fx_param_int(module, "order", 7), dimension);
        }
        else if(dimension_ <= 3) {
          ka_.sea_.Init(fx_param_int(module, "order", 3), dimension);
        }
        else if(dimension_ <= 5) {
          ka_.sea_.Init(fx_param_int(module, "order", 1), dimension);
        }
        else {
          ka_.sea_.Init(fx_param_int(module, "order", 0), dimension);
        }
      }
    }
  };

  /** coarse result on a region */
  class QPostponed {
  public:

    DRange d_density_range_;
    DRange finite_diff_range_;
    double used_error_;
    int n_pruned_;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(d_density_range_);
      OT_MY_OBJECT(finite_diff_range_);
      OT_MY_OBJECT(used_error_);
      OT_MY_OBJECT(n_pruned_);
    }

  public:

    /** initialize postponed information to zero */
    void Init(const Param& param) {
      d_density_range_.Init(0, 0);
      finite_diff_range_.Init(0, 0);
      used_error_ = 0;
      n_pruned_ = 0;
    }

    void Reset(const Param& param) {
      d_density_range_.Init(0, 0);
      finite_diff_range_.Init(0, 0);
      used_error_ = 0;
      n_pruned_ = 0;
    }

    /** accumulate postponed information passed down from above */
    void ApplyPostponed(const Param& param, const QPostponed& other) {
      d_density_range_ += other.d_density_range_;
      finite_diff_range_ += other.finite_diff_range_;
      used_error_ += other.used_error_;
      n_pruned_ += other.n_pruned_;
    }
  };

  /**
   * Coarse result on a region.
   */
  class Delta {
  public:

    /** holds the squared distance bound */
    DRange dsqd_range_;

    /** Density update to apply to children's bound */
    DRange d_density_range_;

    OT_DEF_BASIC(Delta) {
      OT_MY_OBJECT(dsqd_range_);
      OT_MY_OBJECT(d_density_range_);
    }

  public:
    void Init(const Param& param) {
      dsqd_range_.Init(0, 0);
      d_density_range_.Init(0, 0);
    }
  };

  /** individual query result */
  class QResult {
  public:
    DRange density_range_;

    double density_estimate_;

    /** amount of used absolute error for this query point */
    double used_error_;

    /** number of reference points taken care for this query point */
    index_t n_pruned_;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(density_range_);
      OT_MY_OBJECT(density_estimate_);
      OT_MY_OBJECT(used_error_);
      OT_MY_OBJECT(n_pruned_);
    }

  public:
    void Init(const Param& param) {
      density_range_.Init(0, 0);
      density_estimate_ = 0;
      used_error_ = 0;
      n_pruned_ = 0;
    }
    
    /** divide each density by the normalization constant */
    void Postprocess(const Param& param) {
      density_range_ *= param.mul_constant_;
      density_estimate_ *= param.mul_constant_;
    }

    /** apply left over postponed contributions */
    void ApplyPostponed(const Param& param, const QPostponed& postponed) {
      density_range_ += postponed.d_density_range_;
      density_estimate_ += 0.5 * (postponed.finite_diff_range_.lo +
				  postponed.finite_diff_range_.hi);
      used_error_ += postponed.used_error_;
      n_pruned_ += postponed.n_pruned_;
    }
  };

 private:

  class QSummaryResult {
  public:
    
    /** bound on the density from leaves */
    DRange density_range_;
    
    /** maximum amount of error incurred among the query points */
    double used_error_;

    /**
     * minimum bound on the portion of the reference dataset that has been
     * taken care of.
     */
    index_t n_pruned_;

    OT_DEF_BASIC(QSummaryResult) {
      OT_MY_OBJECT(density_range_);
      OT_MY_OBJECT(used_error_);
      OT_MY_OBJECT(n_pruned_);
    }

  public:

    /** initialize summary result to zeros */
    void Init(const Param& param) {
      density_range_.Init(0, 0);
      used_error_ = 0;
      n_pruned_ = 0;
    }

    void StartReaccumulate(const Param& param, Tree *q_node) {
      density_range_.InitEmptySet();
      used_error_ = 0;
      n_pruned_ = param.reference_count_;
    }

    /**
     * refine query summary results by incorporating the given current
     * query result
     */
    void Accumulate(const Param& param, const QResult& result) {
      density_range_ |= result.density_range_;
      used_error_ = max(used_error_, result.used_error_);
      n_pruned_ = min(n_pruned_, result.n_pruned_);
    }

    /**
     * this is the vertical operator that refines the current query summary
     * results based on the summary results owned by the given child
     */
    void Accumulate(const Param& param,
                    const QSummaryResult& result, index_t n_points) {
      density_range_ |= result.density_range_;
      used_error_ += result.used_error_;
      n_pruned_ = min(n_pruned_, result.n_pruned_);
    }

    void FinishReaccumulate(const Param& param, Tree *q_node) {
    }

    /**
     * horizontal join operator that accumulates the current best guess
     * on the density bound on the reference portion that has not been
     * visited so far.
     */
    void ApplySummaryResult(const Param& param,
                            const QSummaryResult& summary_result) {
      density_range_ += summary_result.density_range_;
      used_error_ += summary_result.used_error_;
      n_pruned_ += summary_result.n_pruned_;
    }

    /** apply deltas */
    void ApplyDelta(const Param& param, const Delta& delta) {
      density_range_ += delta.d_density_range_;
    }

    /** apply postponed contributions that were passed down */
    void ApplyPostponed(const Param& param,
                        const QPostponed& postponed, Tree *q_node) {
      density_range_ += postponed.d_density_range_;
      used_error_ += postponed.used_error_;
      n_pruned_ += postponed.n_pruned_;
    }
  };

  class KdeStat {
  public:

    /**
     * summary result
     */
    QSummaryResult summary_result_;

    /**
     * postponed result
     */
    QPostponed postponed_;
    
    /**
     * Far field expansion created by the reference points in this node.
     */
    typename TKernelAux::TFarFieldExpansion farfield_expansion_;
    
    /**
     * Local expansion stored in this node.
     */
    typename TKernelAux::TLocalExpansion local_expansion_;
    
    /** Initialize the statistics */
    void Init(const Param& param) {
      summary_result_.Init(param);
    }

    void Init() {
    }

    void Init(const TKernelAux &ka) {
      farfield_expansion_.Init(ka);
      local_expansion_.Init(ka);
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
      Init();
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const KdeStat& left_stat,
	      const KdeStat& right_stat) {
      Init();
    }
    
    void Init(const Vector& center, const TKernelAux &ka) {
      
      farfield_expansion_.Init(center, ka);
      local_expansion_.Init(center, ka);
      Init();
    }
    
    KdeStat() { }
    
    ~KdeStat() {}
    
  };

  /** parameter list */
  Param parameters_;

  /** query dataset */
  Matrix qset_;

  /** query tree */
  Tree *qroot_;

  /** reference dataset */
  Matrix rset_;
  
  /** reference tree */
  Tree *rroot_;
  
  /** reference weights */
  Vector rset_weights_;

  /** stores results for all query points */
  ArrayList<QResult> q_results_;

  int num_farfield_to_local_prunes_;

  int num_farfield_prunes_;
  
  int num_local_prunes_;
  
  int num_finite_difference_prunes_;

  // preprocessing: scaling the dataset; this has to be moved to the dataset
  // module
  /* scales each attribute to 0-1 using the min/max values */
  void scale_data_by_minmax() {

    int num_dims = rset_.n_rows();
    DHrectBound<2> qset_bound;
    DHrectBound<2> rset_bound;
    qset_bound.Init(qset_.n_rows());
    rset_bound.Init(qset_.n_rows());

    // go through each query/reference point to find out the bounds
    for(index_t r = 0; r < rset_.n_cols(); r++) {
      Vector ref_vector;
      rset_.MakeColumnVector(r, &ref_vector);
      rset_bound |= ref_vector;
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      Vector query_vector;
      qset_.MakeColumnVector(q, &query_vector);
      qset_bound |= query_vector;
    }

    for(index_t i = 0; i < num_dims; i++) {
      DRange qset_range = qset_bound.get(i);
      DRange rset_range = rset_bound.get(i);
      double min_coord = min(qset_range.lo, rset_range.lo);
      double max_coord = max(qset_range.hi, rset_range.hi);
      double width = max_coord - min_coord;

      printf("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);

      for(index_t j = 0; j < rset_.n_cols(); j++) {
	rset_.set(i, j, (rset_.get(i, j) - min_coord) / width);
      }
      
      if(strcmp(fx_param_str(NULL, "query", NULL), 
		fx_param_str_req(NULL, "data"))) {
	for(index_t j = 0; j < qset_.n_cols(); j++) {
	  qset_.set(i, j, (qset_.get(i, j) - min_coord) / width);
	}
      }
    }
  }

  /** exhaustive base KDE case */
  void FKdeBase(Tree *qnode, Tree *rnode) {

    // clear out summary result of the query node so that it can be built up
    // from scratch
    qnode->stat().summary_result_.StartReaccumulate(parameters_, qnode);

    // compute unnormalized sum
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      
      // incorporate the postponed information
      q_results_[q].ApplyPostponed(parameters_, qnode->stat().postponed_);

      // get query point
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = rnode->begin(); r < rnode->end(); r++) {

	// get reference point
	const double *r_col = rset_.GetColumnPtr(r);

	// pairwise distance and kernel value
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double ker_value = parameters_.ka_.kernel_.EvalUnnormOnSq(dsqd);

	// apply exhaustively computed value to the lower/upper bounds and
	// actual density estimate that is to be returned.
	q_results_[q].density_range_ += ker_value;
	q_results_[q].density_estimate_ += ker_value;
      }
      qnode->stat().summary_result_.Accumulate(parameters_, q_results_[q]);
    } // end of looping over each query point

    // finish refining summary result
    qnode->stat().summary_result_.FinishReaccumulate(parameters_, qnode);

    // clear postponed information
    qnode->stat().postponed_.Reset(parameters_);
  }

  bool IntrinsicPrunable(Tree *qnode, Tree *rnode, const Delta &parent_delta,
			 Delta *delta) {

    // compute distance bound between two nodes
    delta->dsqd_range_ = qnode->bound().RangeDistanceSq(rnode->bound());
    
    // compute the bound on kernel contribution
    delta->d_density_range_ = 
      parameters_.ka_.kernel_.RangeUnnormOnSq(delta->dsqd_range_);
    delta->d_density_range_ *= rnode->count();
    
    if(likely(delta->d_density_range_.hi != 0)) {
      return false;
    }
    // if the highest kernel value is zero, then perform exclusion pruning
    else {
      qnode->stat().postponed_.n_pruned_ += rnode->count();
      num_finite_difference_prunes_++;
      return true;
    }
  }
  
  /** 
   * checking for prunbability of the query and the reference pair using
   * series expansion
   */
  bool ExtrinsicPrunableSeriesExpansion
    (Tree *qnode, Tree *rnode, const Delta& delta, 
     const QSummaryResult& q_summary_result) {

    // order of approximation
    int order_farfield_to_local = -1;
    int order_farfield = -1;
    int order_local = -1;

    // actual amount of error incurred per each query/ref pair
    double actual_err_farfield_to_local = 0;
    double actual_err_farfield = 0;
    double actual_err_local = 0;

    // estimated computational cost
    int cost_farfield_to_local = MAXINT;
    int cost_farfield = MAXINT;
    int cost_local = MAXINT;
    int cost_exhaustive = (qnode->count()) * (rnode->count()) * 
      parameters_.dimension_;
    int min_cost = 0;

    // allocated error per each reference point
    double allowed_err =
      (parameters_.relative_error_ * q_summary_result.density_range_.lo -
       q_summary_result.used_error_) / 
      (parameters_.reference_count_ - q_summary_result.n_pruned_);

    // get the order of approximations
    order_farfield_to_local = 
      rnode->stat().farfield_expansion_.OrderForConvertingToLocal
      (rnode->bound(), qnode->bound(), 
       delta.dsqd_range_.lo, delta.dsqd_range_.hi,
       allowed_err, &actual_err_farfield_to_local);
    order_farfield = 
      rnode->stat().farfield_expansion_.OrderForEvaluating
      (rnode->bound(), qnode->bound(), delta.dsqd_range_.lo, 
       delta.dsqd_range_.hi, allowed_err, &actual_err_farfield);
    order_local = 
      qnode->stat().local_expansion_.OrderForEvaluating
      (rnode->bound(), qnode->bound(), delta.dsqd_range_.lo, 
       delta.dsqd_range_.hi, allowed_err, &actual_err_local);

    // update computational cost and compute the minimum
    if(order_farfield_to_local >= 0) {
      cost_farfield_to_local = (int) pow(order_farfield_to_local + 1, 
					 2 * parameters_.dimension_);
    }
    if(order_farfield >= 0) {
      cost_farfield = (int) pow(order_farfield + 1, parameters_.dimension_) * 
	(qnode->count());
    }
    if(order_local >= 0) {
      cost_local = (int) pow(order_local + 1, parameters_.dimension_) * 
	(rnode->count());
    }

    min_cost = min(cost_farfield_to_local, 
		   min(cost_farfield, min(cost_local, cost_exhaustive)));
    
    if(cost_farfield_to_local == min_cost) {
      qnode->stat().postponed_.d_density_range_ += delta.d_density_range_;
      qnode->stat().postponed_.used_error_ += 
	rnode->count() * actual_err_farfield_to_local;
      qnode->stat().postponed_.n_pruned_ += rnode->count();

      rnode->stat().farfield_expansion_.TranslateToLocal
	(qnode->stat().local_expansion_, order_farfield_to_local);
      num_farfield_to_local_prunes_++;
      return true;
    }

    if(cost_farfield == min_cost) {
      qnode->stat().postponed_.d_density_range_ += delta.d_density_range_;
      qnode->stat().postponed_.used_error_ += 
	rnode->count() * actual_err_farfield;
      qnode->stat().postponed_.n_pruned_ += rnode->count();

      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	q_results_[q].density_estimate_ += 
	  rnode->stat().farfield_expansion_.EvaluateField(qset_, q, 
							  order_farfield);
      }
      num_farfield_prunes_++;
      return true;
    }

    if(cost_local == min_cost) {
      qnode->stat().postponed_.d_density_range_ += delta.d_density_range_;
      qnode->stat().postponed_.used_error_ += 
	rnode->count() * actual_err_local;
      qnode->stat().postponed_.n_pruned_ += rnode->count();

      qnode->stat().local_expansion_.AccumulateCoeffs(rset_, rset_weights_,
						      rnode->begin(), 
						      rnode->end(),
						      order_local);
      num_local_prunes_++;
      return true;
    }
    return false;
  }
 
  /** checking for prunability of the query and the reference pair */
  bool ExtrinsicPrunable(Tree *qnode, Tree *rnode, const Delta& delta,
			 const QSummaryResult& q_summary_result) {
    
    double allocated_error =
      (parameters_.relative_error_ * q_summary_result.density_range_.lo -
       q_summary_result.used_error_) * rnode->count() / 
      (parameters_.reference_count_ - q_summary_result.n_pruned_);
    
    // finite difference first
    if(delta.d_density_range_.width() / 2.0 <= allocated_error) {
      qnode->stat().postponed_.d_density_range_ += delta.d_density_range_;
      qnode->stat().postponed_.finite_diff_range_ += delta.d_density_range_;
      qnode->stat().postponed_.used_error_ += 
	delta.d_density_range_.width() / 2.0;
      qnode->stat().postponed_.n_pruned_ += rnode->count();
      num_finite_difference_prunes_++;
      return true;
    }
    // series expansion
    else {
      return ExtrinsicPrunableSeriesExpansion(qnode, rnode, delta,
					      q_summary_result);
    }
  }

  double Heuristic(Tree *qnode, Tree *rnode) {
    return rnode->bound().MinToMidSq(qnode->bound());
  }
  
  /** canonical fast KDE case */
  void FKde(Tree *qnode, Tree *rnode, const Delta& delta,
	    const QSummaryResult& unvisited) {

    // begin prune checks
    QSummaryResult mu(qnode->stat().summary_result_);
    mu.ApplyPostponed(parameters_, qnode->stat().postponed_, qnode);
    mu.ApplySummaryResult(parameters_, unvisited);
    mu.ApplyDelta(parameters_, delta);

    // extrinsic pruning based on lower bound on density estimates
    if(ExtrinsicPrunable(qnode, rnode, delta, mu)) {
      return;
    }

    else { // in case pruning fails
      
      // for leaf pairs, go exhaustive
      if(qnode->is_leaf() && rnode->is_leaf()) {
	FKdeBase(qnode, rnode);
	return;
      }
    
      // if the reference node is a leaf or the query node has more points
      // and it is not a leaf, then split query side.
      else if(rnode->is_leaf() || 
	      (qnode->count() >= rnode->count() && !qnode->is_leaf())) {
	qnode->stat().summary_result_.StartReaccumulate(parameters_, qnode);
	
	// explore the left child
	Tree *left_child = qnode->left();
	Delta left_child_delta;	
	left_child_delta.Init(parameters_);
	left_child->stat().postponed_.ApplyPostponed(parameters_,
						     qnode->stat().postponed_);
	if(!IntrinsicPrunable(left_child, rnode, delta, &left_child_delta)) {
	  FKde(left_child, rnode, left_child_delta, unvisited);
	}

	QSummaryResult tmp_result(left_child->stat().summary_result_);
	tmp_result.ApplyPostponed(parameters_, left_child->stat().postponed_,
				  left_child);
	qnode->stat().summary_result_.Accumulate(parameters_, tmp_result,
						 qnode->count());

	// explore the right child
	Tree *right_child = qnode->right();
	Delta right_child_delta;	
	right_child_delta.Init(parameters_);
	right_child->stat().postponed_.ApplyPostponed
	  (parameters_, qnode->stat().postponed_);
	if(!IntrinsicPrunable(right_child, rnode, delta, &right_child_delta)) {
	  FKde(right_child, rnode, right_child_delta, unvisited);
	}
	tmp_result = right_child->stat().summary_result_;
	tmp_result.ApplyPostponed(parameters_, right_child->stat().postponed_, 
				  right_child);
	qnode->stat().summary_result_.Accumulate(parameters_, tmp_result, 
						 qnode->count());
	
	// finish refining sumamry result for the current query node
	qnode->stat().summary_result_.FinishReaccumulate(parameters_, qnode);

	// clear out postponed information in the current query node
	qnode->stat().postponed_.Reset(parameters_);
	
      } // end of splitting the query case

      // this means, we have to split the reference side
      else {
	Tree *r_left_child = rnode->left();
	Tree *r_right_child = rnode->right();

	Delta delta1, delta2;
	delta1.Init(parameters_);
	delta2.Init(parameters_);
	
	bool intrinsic_prunable_r1 = 
	  IntrinsicPrunable(qnode, r_left_child, delta, &delta1);
	bool intrinsic_prunable_r2 =
	  IntrinsicPrunable(qnode, r_right_child, delta, &delta2);

	if(intrinsic_prunable_r1) {
	  if(!intrinsic_prunable_r2) {
	    FKde(qnode, r_right_child, delta2, unvisited);
	  }
	}
	else if(intrinsic_prunable_r2) {
	  FKde(qnode, r_left_child, delta1, unvisited);
	}
	else {
	  double heur1 = Heuristic(qnode, r_left_child);
	  double heur2 = Heuristic(qnode, r_right_child);
	  
	  if (!(heur1 > heur2)) {
	    QSummaryResult unvisited_for_r1(unvisited);
	    unvisited_for_r1.ApplyDelta(parameters_, delta2);
	    FKde(qnode, r_left_child, delta1, unvisited_for_r1);
	    FKde(qnode, r_right_child, delta2, unvisited);
	  } 
	  else {
	    QSummaryResult unvisited_for_r2(unvisited);
	    unvisited_for_r2.ApplyDelta(parameters_, delta1);
	    FKde(qnode, r_right_child, delta2, unvisited_for_r2);
	    FKde(qnode, r_left_child, delta1, unvisited);
	  }
	}
      }
    } // handling the case in which the extrinsic pruning fails
  }

  /** 
   * pre-processing step - this wouldn't be necessary if the core
   * fastlib supported a Init function for Stat objects that take
   * more arguments.
   */
  void PreProcess(Tree *node) {

    // initialize the center of expansions and bandwidth for
    // series expansion
    node->stat().Init(parameters_.ka_);
    node->bound().CalculateMidpoint
      (node->stat().farfield_expansion_.get_center());
    node->bound().CalculateMidpoint
      (node->stat().local_expansion_.get_center());

    // reset summary result and postponed information
    node->stat().summary_result_.Init(parameters_);
    node->stat().postponed_.Init(parameters_);

    // for non-leaf node, recurse
    if(!node->is_leaf()) {
      PreProcess(node->left());
      PreProcess(node->right());

      // translate multipole moments
      node->stat().farfield_expansion_.TranslateFromFarField
	(node->left()->stat().farfield_expansion_);
      node->stat().farfield_expansion_.TranslateFromFarField
	(node->right()->stat().farfield_expansion_);
    }
    else {
      // exhaustively compute multipole moments
      node->stat().farfield_expansion_.RefineCoeffs
	(rset_, rset_weights_, node->begin(), node->end(), 
	 parameters_.ka_.sea_.get_max_order());
    }
  }

  /** post processing step */
  void PostProcess(Tree *qnode) {

    // for leaf query node, incorporate the postponed info and normalize
    // density estimates
    if(qnode->is_leaf()) {
      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	q_results_[q].ApplyPostponed(parameters_, qnode->stat().postponed_);
	q_results_[q].density_estimate_ +=
	  qnode->stat().local_expansion_.EvaluateField(qset_, q);
	q_results_[q].Postprocess(parameters_);
      }
    }
    // for non-leaf query node,
    else {
      
      // push down approximations and recurse
      qnode->left()->stat().postponed_.ApplyPostponed
	(parameters_, qnode->stat().postponed_);
      qnode->right()->stat().postponed_.ApplyPostponed
	(parameters_, qnode->stat().postponed_);
      
      qnode->stat().local_expansion_.TranslateToLocal
	(qnode->left()->stat().local_expansion_);
      qnode->stat().local_expansion_.TranslateToLocal
	(qnode->right()->stat().local_expansion_);

      PostProcess(qnode->left());      
      PostProcess(qnode->right());
    }
  }

  public:

  // constructor/destructor
  FastKde() {}

  ~FastKde() { 
    
    if(qroot_ != rroot_ ) {
      delete qroot_; 
      delete rroot_; 
    } 
    else {
      delete rroot_;
    }

  }

  // getters and setters

  /** get the reference dataset */
  Matrix &get_reference_dataset() { return rset_; }

  /** get the query dataset */
  Matrix &get_query_dataset() { return qset_; }

  /** get the density estimate */
  void get_density_estimates(Vector *results) { 
    results->Init(q_results_.size());
    
    for(index_t i = 0; i < q_results_.size(); i++) {
      (*results)[i] = q_results_[i].density_estimate_;
    }
  }

  // interesting functions...

  void Compute() {

    num_finite_difference_prunes_ = num_farfield_to_local_prunes_ =
      num_farfield_prunes_ = num_local_prunes_ = 0;

    printf("\nStarting fast KDE...\n");
    fx_timer_start(NULL, "fast_kde_compute");

    // preprocessing step for initializing series expansion objects
    PreProcess(rroot_);
    if(qroot_ != rroot_) {
      PreProcess(qroot_);
    }
    
    // call main routine
    Delta delta, empty_delta;
    empty_delta.Init(parameters_);
    delta.Init(parameters_);
    bool intrinsic_prunable = 
      IntrinsicPrunable(qroot_, rroot_, empty_delta, &delta);

    if(!intrinsic_prunable) {
      QSummaryResult empty_summary_result;
      empty_summary_result.Init(parameters_);
      FKde(qroot_, rroot_, delta, empty_summary_result);
    }

    // postprocessing step for finalizing the sums
    PostProcess(qroot_);
    fx_timer_stop(NULL, "fast_kde_compute");
    printf("\nFast KDE completed...\n");
    printf("Finite difference prunes: %d\n", num_finite_difference_prunes_);
    printf("F2L prunes: %d\n", num_farfield_to_local_prunes_);
    printf("F prunes: %d\n", num_farfield_prunes_);
    printf("L prunes: %d\n", num_local_prunes_);
  }

  void Init() {
    
    Dataset ref_dataset;

    // read in the number of points owned by a leaf
    int leaflen = fx_param_int(NULL, "leaflen", 20);

    // read the datasets
    const char *rfname = fx_param_str_req(NULL, "data");
    const char *qfname = fx_param_str(NULL, "query", rfname);

    // read reference dataset
    ref_dataset.InitFromFile(rfname);
    rset_.Own(&(ref_dataset.matrix()));

    // read the reference weights
    char *rwfname = NULL;
    if(fx_param_exists(NULL, "dwgts")) {
      rwfname = (char *)fx_param_str(NULL, "dwgts", NULL);
    }

    if(rwfname != NULL) {
      Dataset ref_weights;
      ref_weights.InitFromFile(rwfname);
      rset_weights_.Copy(ref_weights.matrix().GetColumnPtr(0),
			 ref_weights.matrix().n_rows());
    }
    else {
      rset_weights_.Init(rset_.n_cols());
      rset_weights_.SetAll(1);
    }

    if(!strcmp(qfname, rfname)) {
      qset_.Alias(rset_);
    }
    else {
      Dataset query_dataset;
      query_dataset.InitFromFile(qfname);
      qset_.Own(&(query_dataset.matrix()));
    }

    // scale dataset if the user wants to
    if(!strcmp(fx_param_str(NULL, "scaling", NULL), "range")) {
      scale_data_by_minmax();
    }

    // construct query and reference trees
    fx_timer_start(NULL, "tree_d");
    rroot_ = tree::MakeKdTreeMidpoint<Tree>(rset_, leaflen);

    if(!strcmp(qfname, rfname)) {
      qroot_ = rroot_;
    }
    else {
      qroot_ = tree::MakeKdTreeMidpoint<Tree>(qset_, leaflen);
    }
    fx_timer_stop(NULL, "tree_d");
    
    // initialize the density lists
    q_results_.Init(qset_.n_cols());
    for(index_t i = 0; i < qset_.n_cols(); i++) {
      q_results_[i].Init(parameters_);
    }

    // initialize parameter list
    parameters_.Init(fx_root);
    parameters_.reference_count_ = rset_.n_cols();
    parameters_.query_count_ = qset_.n_cols();
    parameters_.FinalizeInit(fx_root, rset_.n_rows());
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(NULL, "fast_kde_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", q_results_[q].density_estimate_);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#endif
