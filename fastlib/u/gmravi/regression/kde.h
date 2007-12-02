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

  void ComputeMaximumRelativeError(const Vector &density_estimate) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < densities_.length(); q++) {
      double rel_err = fabs(density_estimate[q] - densities_[q]) / 
	densities_[q];
      
      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    fx_format_result(NULL, "maxium_relative_error_for_fast_KDE", "%g", 
		     max_rel_err);
  }

};


template<typename TKernel, typename TKernelAux>
class FastKde {

  public:

  // forward declaration of KdeStat class
  class KdeStat;

  // our tree type using the KdeStat
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, KdeStat > Tree;
  
  class KdeStat {
  public:
    
    /** lower bound on the densities for the query points owned by this node 
     */
    double mass_l_;
    
    /**
     * additional offset for the lower bound on the densities for the query
     * points owned by this node (for leaf nodes only).
     */
    double more_l_;
    
    /**
     * lower bound offset passed from above
     */
    double owed_l_;
    
    /** stores the portion pruned by finite difference
     */
    double mass_e_;

    /** upper bound on the densities for the query points owned by this node 
     */
    double mass_u_;
    
    /**
     * additional offset for the upper bound on the densities for the query
     * points owned by this node (for leaf nodes only)
     */
    double more_u_;
    
    /**
     * upper bound offset passed from above
     */
    double owed_u_;
    
    /** extra error that can be used for the query points in this node */
    double mass_t_;
    
    /**
     * Far field expansion created by the reference points in this node.
     */
    typename TKernelAux::TFarFieldExpansion farfield_expansion_;
    
    /**
     * Local expansion stored in this node.
     */
    typename TKernelAux::TLocalExpansion local_expansion_;
    
    /** Initialize the statistics */
    void Init() {
      mass_l_ = 0;
      more_l_ = 0;
      owed_l_ = 0;
      mass_e_ = 0;
      mass_u_ = 0;
      more_u_ = 0;
      owed_u_ = 0;
      mass_t_ = 0;    
    }
    
    void Init(double bandwidth, 
	      typename TKernelAux::TSeriesExpansionAux *sea) {

      farfield_expansion_.Init(bandwidth, sea);
      local_expansion_.Init(bandwidth, sea);
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
      Init();
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const KdeStat& left_stat,
	      const KdeStat& right_stat) {
      Init();
    }
    
    void Init(double bandwidth, const Vector& center,
	      typename TKernelAux::TSeriesExpansionAux *sea) {
      
      farfield_expansion_.Init(bandwidth, center, sea);
      local_expansion_.Init(bandwidth, center, sea);
      Init();
    }

    void MergeChildBounds(KdeStat &left_stat, KdeStat &right_stat) {

      // steal left and right children's tokens
      double min_mass_t = min(left_stat.mass_t_, right_stat.mass_t_);

      // improve lower and upper bound
      mass_l_ = max(mass_l_, min(left_stat.mass_l_, right_stat.mass_l_));
      mass_u_ = min(mass_u_, max(left_stat.mass_u_, right_stat.mass_u_));
      mass_t_ += min_mass_t;
      left_stat.mass_t_ -= min_mass_t;
      right_stat.mass_t_ -= min_mass_t;
    }

    void PushDownTokens
      (KdeStat &left_stat, KdeStat &right_stat, double *de,
       typename TKernelAux::TLocalExpansion *local_expansion, double *dt) {
    
      if(de != NULL) {
	double de_ref = *de;
	left_stat.mass_e_ += de_ref;
	right_stat.mass_e_ += de_ref;
	*de = 0;
      }

      if(local_expansion != NULL) {
	local_expansion->TranslateToLocal(left_stat.local_expansion_);
	local_expansion->TranslateToLocal(right_stat.local_expansion_);
      }
      if(dt != NULL) {
	double dt_ref = *dt;
	left_stat.mass_t_ += dt_ref;
	right_stat.mass_t_ += dt_ref;
	*dt = 0;
      }
    }

    KdeStat() { }
    
    ~KdeStat() {}
    
  };
  
  private:

  /** series expansion auxililary object */
  typename TKernelAux::TSeriesExpansionAux sea_;

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

  /** list of kernels to evaluate */
  TKernel kernel_;

  /** lower bound on the densities */
  Vector densities_l_;

  /** densities computed */
  Vector densities_e_;

  /** upper bound on the densities */
  Vector densities_u_;

  /** accuracy parameter */
  double tau_;

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

  // member functions
  void UpdateBounds(Tree *qnode, Tree *rnode, 
		    double *dl, double *de, double *du, double *dt,
		    int *order_farfield_to_local, int *order_farfield,
		    int *order_local) {
    
    // query self statistics
    KdeStat &qstat = qnode->stat();

    // reference node statistics
    KdeStat &rstat = rnode->stat();

    // incorporate into the self
    double dl_ref = *dl;
    double du_ref = *du;
    qstat.mass_l_ += dl_ref;
    qstat.mass_u_ += du_ref;

    // incorporate finite difference pruning if available
    if(de != NULL) {
      qstat.mass_e_ += (*de);
    }

    // incorporate token change
    if(dt != NULL) {
      qstat.mass_t_ += (*dt);
    }
    
    // incorporate series approximation into the self

    // far field to local translation
    if(order_farfield_to_local != NULL && *order_farfield_to_local >= 0) {
      rstat.farfield_expansion_.TranslateToLocal(qstat.local_expansion_,
						 *order_farfield_to_local);
    }
    // far field pruning
    else if(order_farfield != NULL && *order_farfield >= 0) {
      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	densities_e_[q] += 
	  rstat.farfield_expansion_.EvaluateField(&qset_, q, NULL, 
						  *order_farfield);
      }
    }
    // local accumulation pruning
    else if(order_local != NULL && *order_local >= 0) {
      qstat.local_expansion_.AccumulateCoeffs(rset_, rset_weights_,
					      rnode->begin(), rnode->end(),
					      *order_local);
    }
    
    // for a leaf node, incorporate the lower and upper bound changes into
    // its additional offset
    if(qnode->is_leaf()) 
      {
	qstat.more_l_ += dl_ref;
	qstat.more_u_ += du_ref;
      } 
    
    // otherwise, incorporate the bound changes into the owed slots of
    // the immediate descendants
    else {      
      qnode->left()->stat().owed_l_ += dl_ref; //transmission of the owed valuesto the children
      qnode->left()->stat().owed_u_ += du_ref;
      qnode->right()->stat().owed_l_ += dl_ref;
      qnode->right()->stat().owed_u_ += du_ref;
    }
  }

  /** exhaustive base KDE case */
  void FKdeBase(Tree *qnode, Tree *rnode) {

    // compute unnormalized sum
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      
      // get query point
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = rnode->begin(); r < rnode->end(); r++) {

	// get reference point
	const double *r_col = rset_.GetColumnPtr(r);

	// pairwise distance and kernel value
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double ker_value = kernel_.EvalUnnormOnSq(dsqd);

	densities_l_[q] += ker_value;
	densities_e_[q] += ker_value;
	densities_u_[q] += ker_value;
      }
    }
    
    // tally up the unused error components due to exhaustive computation
    qnode->stat().mass_t_ += rnode->count();

    // get a tighter lower and upper bound by looping over each query point
    // in the current query leaf node
    double min_l = MAXDOUBLE;
    double max_u = -MAXDOUBLE;
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      if(densities_l_[q] < min_l) {
	min_l = densities_l_[q];
      }
      if(densities_u_[q] > max_u) {
	max_u = densities_u_[q];
      }
    }
    
    // subtract the contribution accounted by the exhaustive computation
    qnode->stat().more_u_ -= rnode->count();

    // tighten lower and upper bound
    qnode->stat().mass_l_ = min_l + qnode->stat().more_l_;
    qnode->stat().mass_u_ = max_u + qnode->stat().more_u_;
  }

  /** 
   * checking for prunability of the query and the reference pair using
   * four types of pruning methods
   */
  int PrunableEnhanced(Tree *qnode, Tree *rnode, DRange &dsqd_range,
		       DRange &kernel_value_range, double &dl, double &du, 
		       double &dt, int &order_farfield_to_local,
		       int &order_farfield, int &order_local) {

    int dim = rset_.n_rows();

    // actual amount of error incurred per each query/ref pair
    double actual_err_farfield_to_local = 0;
    double actual_err_farfield = 0;
    double actual_err_local = 0;

    // estimated computational cost
    int cost_farfield_to_local = MAXINT;
    int cost_farfield = MAXINT;
    int cost_local = MAXINT;
    int cost_exhaustive = (qnode->count()) * (rnode->count()) * dim;
    int min_cost = 0;

    // query node and reference node statistics
    KdeStat &qstat = qnode->stat();
    KdeStat &rstat = rnode->stat();
    
    // expansion objects
    typename TKernelAux::TFarFieldExpansion &farfield_expansion = 
      rstat.farfield_expansion_;
    typename TKernelAux::TLocalExpansion &local_expansion = 
      qstat.local_expansion_;

    // number of reference points
    int num_references = rnode->count();

    // try pruning after bound refinement:
    // the new lower bound after incorporating new info
    dl = kernel_value_range.lo * num_references;
    du = -kernel_value_range.hi * num_references;

    // refine the lower bound using the new lower bound info
    double new_mass_l = qstat.mass_l_ + dl;    
    double allowed_err = tau_ * new_mass_l *
      ((double)(num_references + qstat.mass_t_)) / 
      ((double) rroot_->count() * num_references);
    
    // get the order of approximations
    order_farfield_to_local = 
      farfield_expansion.OrderForConvertingToLocal
      (rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi, 
       allowed_err, &actual_err_farfield_to_local);
    order_farfield = 
      farfield_expansion.OrderForEvaluating(rnode->bound(), qnode->bound(),
					    dsqd_range.lo, dsqd_range.hi,
					    allowed_err, &actual_err_farfield);
    order_local = 
      local_expansion.OrderForEvaluating(rnode->bound(), qnode->bound(), 
					 dsqd_range.lo, dsqd_range.hi,
					 allowed_err, &actual_err_local);

    // update computational cost and compute the minimum
    if(order_farfield_to_local >= 0) {
      cost_farfield_to_local = (int) pow(order_farfield_to_local + 1, 
					 2 * dim);
    }
    if(order_farfield >= 0) {
      cost_farfield = (int) pow(order_farfield + 1, dim) * (qnode->count());
    }
    if(order_local >= 0) {
      cost_local = (int) pow(order_local + 1, dim) * (rnode->count());
    }

    min_cost = min(cost_farfield_to_local, 
		   min(cost_farfield, min(cost_local, cost_exhaustive)));

    if(cost_farfield_to_local == min_cost) {
      dt = num_references * 
	(1.0 - (rroot_->count()) * actual_err_farfield_to_local / 
	 (new_mass_l * tau_));
      order_farfield = order_local = -1;
      num_farfield_to_local_prunes_++;
      return 1;
    }

    if(cost_farfield == min_cost) {
      dt = num_references * 
	(1.0 - (rroot_->count()) * actual_err_farfield / (new_mass_l * tau_));
      order_farfield_to_local = order_local = -1;
      num_farfield_prunes_++;
      return 1;
    }

    if(cost_local == min_cost) {
      dt = num_references * 
	(1.0 - (rroot_->count()) * actual_err_local / (new_mass_l * tau_));
      order_farfield_to_local = order_farfield = -1;
      num_local_prunes_++;
      return 1;
    }

    order_farfield_to_local = order_farfield = order_local = -1;
    dl = du = dt = 0;
    return 0;
  }

  /** checking for prunability of the query and the reference pair */
  int Prunable(Tree *qnode, Tree *rnode, DRange &dsqd_range,
	       DRange &kernel_value_range, double &dl, double &de,
	       double &du, double &dt) {

    // query node stat
    KdeStat &stat = qnode->stat();
    
    // number of reference points
    int num_references = rnode->count();

    // try pruning after bound refinement: first compute distance/kernel
    // value bounds
    dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
    dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
    kernel_value_range = kernel_.RangeUnnormOnSq(dsqd_range);

    // the new lower bound after incorporating new info
    dl = kernel_value_range.lo * num_references;
    de = 0.5 * num_references * 
      (kernel_value_range.lo + kernel_value_range.hi);
    du = -kernel_value_range.hi * num_references;

    // refine the lower bound using the new lower bound info
    double new_mass_l = stat.mass_l_ + dl;    
    double allowed_err = tau_ * new_mass_l *
      ((double)(num_references + stat.mass_t_)) / ((double) rroot_->count());

    // this is error per each query/reference pair for a fixed query
    double m = 0.5 * (kernel_value_range.hi - kernel_value_range.lo);

    // this is total error for each query point
    double error = m * num_references;

    // check pruning condition
    if(error <= allowed_err) {
      dt = num_references * 
	(1.0 - (rroot_->count()) * m / (new_mass_l * tau_));
      num_finite_difference_prunes_++;
      return 1;
    }
    else {
      dl = de = du = dt = 0;
      return 0;
    }
  }

  /** determine which of the node to expand first */
  void BestNodePartners(Tree *nd, Tree *nd1, Tree *nd2, Tree **partner1,
			Tree **partner2) {
    
    double d1 = nd->bound().MinDistanceSq(nd1->bound());
    double d2 = nd->bound().MinDistanceSq(nd2->bound());

    if(d1 <= d2) {
      *partner1 = nd1;
      *partner2 = nd2;
    }
    else {
      *partner1 = nd2;
      *partner2 = nd1;
    }
  }

  /** canonical fast KDE case */
  void FKde(Tree *qnode, Tree *rnode) {

    /** temporary variable for storing lower bound change */
    double dl = 0, de = 0, du = 0, dt = 0;
    int order_farfield_to_local = -1, order_farfield = -1, order_local = -1;

    // temporary variable for holding distance/kernel value bounds
    DRange dsqd_range;
    DRange kernel_value_range;

    // query node statistics
    KdeStat &stat = qnode->stat();

    // left child and right child of query node statistics
    KdeStat *left_stat = NULL;
    KdeStat *right_stat = NULL;

    // process density bound changes sent from the ancestor query nodes,
    UpdateBounds(qnode, rnode, &stat.owed_l_, NULL, &stat.owed_u_, NULL,
		 NULL, NULL, NULL);

    // for non-leaf query node, tighten lower/upper bounds and the 
    // reclaim tokens unused by the children.
    if(!qnode->is_leaf()) {
      left_stat = &(qnode->left()->stat());
      right_stat = &(qnode->right()->stat());
      stat.MergeChildBounds(*left_stat, *right_stat);
    }

    // try finite difference pruning first
    if(Prunable(qnode, rnode, dsqd_range, kernel_value_range, 
		dl, de, du, dt)) {
      UpdateBounds(qnode, rnode, &dl, &de, &du, &dt, NULL, NULL, NULL);
      return;
    }
    
    // try series-expansion pruning
    else if(PrunableEnhanced(qnode, rnode, dsqd_range, kernel_value_range, 
			     dl, du, dt, order_farfield_to_local, 
			     order_farfield, order_local)) {

      UpdateBounds(qnode, rnode, &dl, NULL, &du, &dt,
		   &order_farfield_to_local, &order_farfield,
		   &order_local);
      return;
    }

    // for leaf query node
    if(qnode->is_leaf()) {
      
      // for leaf pairs, go exhaustive
      if(rnode->is_leaf()) {
	FKdeBase(qnode, rnode);
	return;
      }
      
      // for non-leaf reference, expand reference node
      else {
	Tree *rnode_first = NULL, *rnode_second = NULL;
	BestNodePartners(qnode, rnode->left(), rnode->right(), &rnode_first,
			 &rnode_second);
	FKde(qnode, rnode_first);
	FKde(qnode, rnode_second);
	return;
      }
    }
    
    // for non-leaf query node
    else {
      
      // for a leaf reference node, expand query node
      if(rnode->is_leaf()) {
	Tree *qnode_first = NULL, *qnode_second = NULL;

	stat.PushDownTokens(*left_stat, *right_stat, NULL, NULL, 
			    &stat.mass_t_);
	BestNodePartners(rnode, qnode->left(), qnode->right(), &qnode_first,
			 &qnode_second);
	FKde(qnode_first, rnode);
	FKde(qnode_second, rnode);
	return;
      }

      // for non-leaf reference node, expand both query and reference nodes
      else {
	Tree *rnode_first = NULL, *rnode_second = NULL;
	stat.PushDownTokens(*left_stat, *right_stat, NULL, NULL, 
			    &stat.mass_t_);
	
	BestNodePartners(qnode->left(), rnode->left(), rnode->right(),
			 &rnode_first, &rnode_second);
	FKde(qnode->left(), rnode_first);
	FKde(qnode->left(), rnode_second);

	BestNodePartners(qnode->right(), rnode->left(), rnode->right(),
			 &rnode_first, &rnode_second);
	FKde(qnode->right(), rnode_first);
	FKde(qnode->right(), rnode_second);
	return;
      }
    }
  }

  /** 
   * pre-processing step - this wouldn't be necessary if the core
   * fastlib supported a Init function for Stat objects that take
   * more arguments.
   */
  void PreProcess(Tree *node) {

    // initialize the center of expansions and bandwidth for
    // series expansion
    node->stat().Init(sqrt(kernel_.bandwidth_sq()), &sea_);
    node->bound().CalculateMidpoint
      (node->stat().farfield_expansion_.get_center());
    node->bound().CalculateMidpoint
      (node->stat().local_expansion_.get_center());
    
    // initialize lower bound to 0
    node->stat().mass_l_ = 0;
    
    // set the finite difference approximated amounts to 0
    node->stat().mass_e_ = 0;

    // set the upper bound to the number of reference points
    node->stat().mass_u_ = rset_.n_cols();

    // set the number of tokens to 0
    node->stat().mass_t_ = 0;

    // for non-leaf node, recurse
    if(!node->is_leaf()) {
      node->stat().owed_l_ = node->stat().owed_u_ = 0;
      PreProcess(node->left());
      PreProcess(node->right());

      // translate multipole moments
      node->stat().farfield_expansion_.TranslateFromFarField
	(node->left()->stat().farfield_expansion_);
      node->stat().farfield_expansion_.TranslateFromFarField
	(node->right()->stat().farfield_expansion_);
    }
    else {
      node->stat().more_l_ = node->stat().more_u_ = 0;
      
      // exhaustively compute multipole moments
      node->stat().farfield_expansion_.RefineCoeffs(rset_, rset_weights_,
						    node->begin(), node->end(),
						    sea_.get_max_order());
    }
  }

  /** post processing step */
  void PostProcess(Tree *qnode) {
    
    KdeStat &stat = qnode->stat();

    // for leaf query node
    if(qnode->is_leaf()) {
      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	densities_e_[q] +=
	  stat.local_expansion_.EvaluateField(&qset_, q, NULL) +
	  stat.mass_e_;
      }
    }
    else {

      // push down approximations
      stat.PushDownTokens(qnode->left()->stat(), qnode->right()->stat(),
			  &stat.mass_e_, &stat.local_expansion_, NULL);
      PostProcess(qnode->left());
      PostProcess(qnode->right());
    }
  }

  void NormalizeDensities() {
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows()) *
      rset_.n_cols();
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_l_[q] /= norm_const;
      densities_e_[q] /= norm_const;
      densities_u_[q] /= norm_const;
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
  const Vector &get_density_estimates() { return densities_e_; }

  // interesting functions...

  void Compute(double tau) {

    // set accuracy parameter
    tau_ = tau;
    
    // initialize the lower and upper bound densities
    densities_l_.SetZero();
    densities_e_.SetZero();
    densities_u_.SetAll(rset_.n_cols());

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
    FKde(qroot_, rroot_);

    // postprocessing step for finalizing the sums
    PostProcess(qroot_);

    // normalize densities
    NormalizeDensities();
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
    densities_l_.Init(qset_.n_cols());
    densities_e_.Init(qset_.n_cols());
    densities_u_.Init(qset_.n_cols());

    // initialize the kernel
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));

    // initialize the series expansion object
    if(qset_.n_rows() <= 2) {
      sea_.Init(fx_param_int(NULL, "order", 5), qset_.n_rows());
    }
    else {
      sea_.Init(fx_param_int(NULL, "order", 0), qset_.n_rows());
    }
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(NULL, "fast_kde_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", densities_e_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#endif
