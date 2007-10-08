#ifndef KDE_H
#define KDE_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/kernel_derivative.h"

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
  }

  void Init() {
    densities_.SetZero();
  }

  void Init(Matrix &qset, Matrix &rset) {

    // get datasets
    qset_.Own(&qset);
    rset_.Own(&rset);

    // get bandwidth
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));
    
    // allocate density storage
    densities_.Init(qset.n_cols());
    densities_.SetZero();    
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(NULL, "naive_kde_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    densities_.PrintDebug("Naive KDE results", stream);
    
    if(stream != stdout) {
      fclose(stream);
    }    
  }

  void ComputeMaximumRelativeError(const Vector &density_estimate) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < densities_.length(); q++) {
      double rel_err = (density_estimate[q] - densities_[q]) / densities_[q];
      
      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    fx_format_result(NULL, "maxium_relative_error_for_fast_KDE", "%g", 
		     max_rel_err);
  }

};

template<typename TKernel, typename TKernelDerivative>
class KdeStat {
 public:
    
  /** lower bound on the densities for the query points owned by this node */
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
    
  /** accumulated density estimates for the query points owned by this node */
  double mass_e_;
    
  /** upper bound on the densities for the query points owned by this node */
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
  FarFieldExpansion<TKernel, TKernelDerivative> farfield_expansion_;
    
  /**
   * Local expansion stored in this node.
   */
  LocalExpansion<TKernel, TKernelDerivative> local_expansion_;
    
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
    
  void Init(double bandwidth, SeriesExpansionAux *sea) {
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
	    SeriesExpansionAux *sea) {
      
    farfield_expansion_.Init(bandwidth, center, sea);
    local_expansion_.Init(bandwidth, center, sea);
    Init();
  }
    
  void UpdateBounds(KdeStat *left_stat, KdeStat *right_stat, 
		    double *dl, double *de, double *du, double *dt) {
      
    // incorporate into the self
    double dl_ref = *dl;
    double du_ref = *du;
    mass_l_ += dl_ref;

    if(de != NULL) {
      mass_e_ += (*de);
      *de = 0;
    }

    mass_u_ += du_ref;
    
    if(dt != NULL) {
      mass_t_ += (*dt);
      *dt = 0;
    }
      
    // for a leaf node, incorporate the lower and upper bound changes into
    // its additional offset
    if(left_stat == NULL) {
      more_l_ += dl_ref;
      more_u_ += du_ref;
    } 
      
    // otherwise, incorporate the bound changes into the owed slots of
    // the immediate descendants
    else {
	
      left_stat->owed_l_ += dl_ref;
      left_stat->owed_u_ += du_ref;
      right_stat->owed_l_ += dl_ref;
      right_stat->owed_u_ += du_ref;
    }
    
    *dl = 0;
    *du = 0;
  }

  void MergeChildBounds(KdeStat &left_stat, KdeStat &right_stat) {
    double min_mass_t = min(left_stat.mass_t_, right_stat.mass_t_);

    mass_l_ = max(mass_l_, min(left_stat.mass_l_, right_stat.mass_l_));
    mass_u_ = min(mass_u_, max(left_stat.mass_u_, right_stat.mass_u_));
    mass_t_ += min_mass_t;
    left_stat.mass_t_ -= min_mass_t;
    right_stat.mass_t_ -= min_mass_t;
  }

  void PushDownTokens(KdeStat &left_stat, KdeStat &right_stat,
		      double *de, double *dt) {
    
    if(de != NULL) {
      double de_ref = *de;
      left_stat.mass_e_ += de_ref;
      right_stat.mass_e_ += de_ref;
      *de = 0;
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

/** computing kernel estimate using Fast Fourier Transform */
template<typename TKernel>
class FFTKde {

 public:
  
  FFTKde() {}
  
  ~FFTKde() {}

};

template<typename TKernel, typename TKernelDerivative>
class FastKde {
  
 private:

  typedef BinarySpaceTree<DHrectBound<2>, Matrix,
    KdeStat<TKernel, TKernelDerivative> > Tree;

  /** series expansion auxililary object */
  SeriesExpansionAux sea_;

  /** query dataset */
  Matrix qset_;

  /** query tree */
  Tree *qroot_;

  /** reference dataset */
  Matrix rset_;

  /** reference tree */
  Tree *rroot_;

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

  /** temporary variable for storing lower bound change */
  double dl_;
  
  double de_;
  
  double du_;
  
  double dt_;

  int order_farfield_;
  
  int order_local_;
  
  int order_farfield_to_local_;

  // member functions

  /** exhaustive base KDE case */
  void FKdeBase(Tree *qnode, Tree *rnode) {

    // compute unnormalized sum
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = rnode->begin(); r < rnode->end(); r++) {
	const double *r_col = rset_.GetColumnPtr(r);
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double ker_value = kernel_.EvalUnnormOnSq(dsqd);

	densities_l_[q] += ker_value;
	densities_e_[q] += ker_value;
	densities_u_[q] += ker_value;
      }
    }
    
    // tally up the unused error components
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
  int PrunableEnhanced(Tree *qnode, Tree *rnode) {

    // actual amount of error incurred per each query/ref pair
    double actual_err = 0;

    // query node stat
    KdeStat<TKernel, TKernelDerivative> &stat = qnode->stat();
    
    // expansion objects
    FarFieldExpansion<TKernel, TKernelDerivative> &farfield_expansion
      = stat.farfield_expansion_;
    LocalExpansion<TKernel, TKernelDerivative> &local_expansion
      = stat.local_expansion_;

    // number of reference points
    int num_references = rnode->count();
    
    // try pruning after bound refinement:
    DRange dsqd_range;
    dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
    dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
    DRange kernel_value_range = kernel_.RangeUnnormOnSq(dsqd_range);
    
    // the new lower bound after incorporating new info
    dl_ = kernel_value_range.lo * num_references;
    de_ = 0.5 * num_references * 
      (kernel_value_range.lo + kernel_value_range.hi);
    du_ = -kernel_value_range.hi * num_references;

    // refine the lower bound using the new lower bound info
    double new_mass_l = stat.mass_l_ + dl_;    
    double allowed_err = tau_ * new_mass_l *
      ((double)(num_references + stat.mass_t_)) / 
      ((double) rroot_->count() * num_references);
    
    // get the order of approximations
    order_farfield_to_local_ = 
      farfield_expansion.OrderForConvertingtoLocal(rnode->bound(), 
						   qnode->bound(),
						   dsqd_range.lo, allowed_err,
						   &actual_err);
    if(order_farfield_to_local_ >= 0) {
      return 1;
    }
    
    order_farfield_ =
      farfield_expansion.OrderForEvaluating(rnode->bound(), dsqd_range.lo,
					    allowed_err, &actual_err);
    if(order_farfield_ >= 0) {
      return 1;
    }

    order_local_ =
      local_expansion.OrderForEvaluating(qnode->bound(), dsqd_range.lo,
					 allowed_err, &actual_err);

    if(order_local_ >= 0) {
      return 1;
    }

    return 0;
  }

  /** checking for prunability of the query and the reference pair */
  int Prunable(Tree *qnode, Tree *rnode) {

    // query node stat
    KdeStat<TKernel, TKernelDerivative> &stat = qnode->stat();
    
    // number of reference points
    int num_references = rnode->count();

    // try pruning after bound refinement:
    DRange dsqd_range;
    dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
    dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
    DRange kernel_value_range = kernel_.RangeUnnormOnSq(dsqd_range);
    
    // the new lower bound after incorporating new info
    dl_ = kernel_value_range.lo * num_references;
    de_ = 0.5 * num_references * 
      (kernel_value_range.lo + kernel_value_range.hi);
    du_ = -kernel_value_range.hi * num_references;

    // refine the lower bound using the new lower bound info
    double new_mass_l = stat.mass_l_ + dl_;    
    double allowed_err = tau_ * new_mass_l *
      ((double)(num_references + stat.mass_t_)) / ((double) rroot_->count());

    // this is error per each query/reference pair for a fixed query
    double m = 0.5 * (kernel_value_range.hi - kernel_value_range.lo);

    // this is total error for each query point
    double error = m * num_references;

    /*** DEBUGGINGIGNGI! */
    PrunableEnhanced(qnode, rnode);
    /*** END DEBUG!!! */

    // check pruning condition
    if(error <= allowed_err) {
      dt_ = num_references * 
	(1.0 - (rroot_->count()) * m / (new_mass_l * tau_));
      return 1;
    }
    else {
      dl_ = du_ = de_ = dt_ = 0;
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

  /** canonical dualtree KDE case */
  void FKde(Tree *qnode, Tree *rnode) {

    KdeStat<TKernel, TKernelDerivative> &stat = qnode->stat();
    KdeStat<TKernel, TKernelDerivative> *left_stat = NULL;
    KdeStat<TKernel, TKernelDerivative> *right_stat = NULL;

    // process density bound changes sent from the ancestor query nodes
    // then tighten lower/upper bounds and the error reclaimed based on
    // the children
    if(!qnode->is_leaf()) {
      left_stat = &(qnode->left()->stat());
      right_stat = &(qnode->right()->stat());
      stat.UpdateBounds(left_stat, right_stat, &stat.owed_l_, NULL, 
			&stat.owed_u_, NULL);
      stat.MergeChildBounds(*left_stat, *right_stat);
    }

    // if prunable, then prune
    if(Prunable(qnode, rnode)) {
      stat.UpdateBounds(left_stat, right_stat, &dl_, &de_, &du_, &dt_);
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

	stat.PushDownTokens(*left_stat, *right_stat, NULL, &stat.mass_t_);
	BestNodePartners(rnode, qnode->left(), qnode->right(), &qnode_first,
			 &qnode_second);
	FKde(qnode_first, rnode);
	FKde(qnode_second, rnode);
	return;
      }

      // for non-leaf reference node, expand both query and reference nodes
      else {
	Tree *rnode_first = NULL, *rnode_second = NULL;
	stat.PushDownTokens(*left_stat, *right_stat, NULL, &stat.mass_t_);
	
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

    node->stat().Init(sqrt(kernel_.bandwidth_sq()), &sea_);
    node->bound().CalculateMidpoint
      (&(node->stat().farfield_expansion_.get_center()));
    node->bound().CalculateMidpoint
      (&(node->stat().local_expansion_.get_center()));

    // for non-leaf node, recurse
    if(!node->is_leaf()) {
      PreProcess(node->left());
      PreProcess(node->right());
    }
  }

  /** post processing step */
  void PostProcess(Tree *qnode) {
    
    KdeStat<TKernel, TKernelDerivative> &stat = qnode->stat();

    // for leaf query node
    if(qnode->is_leaf()) {
      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	densities_e_[q] += stat.mass_e_;
      }
    }
    else {

      // push down approximations
      stat.PushDownTokens(qnode->left()->stat(), qnode->right()->stat(),
			  &stat.mass_e_, NULL);
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
    densities_u_.SetZero();

    fx_timer_start(NULL, "fast_kde_compute");


    // preprocessing step for initializing series expansion objects
    PreProcess(rroot_);
    if(qroot_ != rroot_) {
      PreProcess(qroot_);
    }
    
    // call main routine
    dl_ = de_ = du_ = dt_ = 0;
    order_farfield_ = order_local_ = order_farfield_to_local_ = -1;
    FKde(qroot_, rroot_);

    // postprocessing step
    PostProcess(qroot_);

    // normalize densities
    NormalizeDensities();
    fx_timer_stop(NULL, "fast_kde_compute");
  }

  void Init() {
    
    Dataset ref_dataset;

    // read in the number of points owned by a leaf
    int leaflen = fx_param_int(NULL, "leaflen", 20);

    // read the datasets
    const char *rfname = fx_param_str_req(NULL, "data");
    const char *qfname = fx_param_str(NULL, "query", rfname);
    
    // construct query and reference trees
    ref_dataset.InitFromFile(rfname);
    rset_.Own(&(ref_dataset.matrix()));

    fx_timer_start(NULL, "tree_d");
    rroot_ = tree::MakeKdTreeMidpoint<Tree>(rset_, leaflen, NULL);

    if(!strcmp(qfname, rfname)) {
      qset_.Alias(rset_);
      qroot_ = rroot_;
    }
    else {
      Dataset query_dataset;
      query_dataset.InitFromFile(qfname);
      qset_.Own(&(query_dataset.matrix()));
      qroot_ = tree::MakeKdTreeMidpoint<Tree>(qset_, leaflen, NULL);
    }
    fx_timer_stop(NULL, "tree_d");

    // initialize the density lists
    densities_l_.Init(qset_.n_cols());
    densities_l_.SetZero();
    densities_e_.Init(qset_.n_cols());
    densities_e_.SetZero();
    densities_u_.Init(qset_.n_cols());
    densities_u_.SetZero();

    // initialize the kernel
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));

    // initialize the series expansion object
    sea_.Init(14, qset_.n_rows());
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(NULL, "fast_kde_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    densities_e_.PrintDebug("Fast KDE results", stream);
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#endif
