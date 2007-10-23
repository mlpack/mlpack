#ifndef LPR_H
#define LPR_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/mult_farfield_expansion.h"
#include "u/dongryel/series_expansion/mult_local_expansion.h"
#include "u/dongryel/series_expansion/kernel_aux.h"

template<typename TKernel>
class NaiveLpr {

 private:
  
  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** reference target */
  Vector rset_targets_;

  /** kernel */
  TKernel kernel_;

  /** numerator vector X^T W(q) Y for each query point */
  ArrayList<Vector> numerator_;
  
  /** denominator matrix X^T W(q) X for each query point */
  ArrayList<Matrix> denominator_;

  /** computed regression values */
  Vector regression_values_;

  /** local polynomial approximation order */
  int lpr_order_;

  /** total number of coefficients for the local polynomial */
  int total_num_coeffs_;

  void PseudoInverse(const Matrix &A, Matrix *A_inv) {
    Vector ro_s;
    Matrix ro_U, ro_VT;

    // compute the SVD of A
    la::SVDInit(A, &ro_s, &ro_U, &ro_VT);
    
    // take the transpose of V^T and U
    Matrix ro_VT_trans;
    Matrix ro_U_trans;
    la::TransposeInit(ro_VT, &ro_VT_trans);
    la::TransposeInit(ro_U, &ro_U_trans);
    Matrix ro_s_inv;
    ro_s_inv.Init(ro_VT_trans.n_cols(), ro_U_trans.n_rows());
    ro_s_inv.SetZero();

    // initialize the diagonal by the inverse of ro_s
    for(index_t i = 0; i < ro_s.length(); i++) {
      ro_s_inv.set(i, i, 1.0 / ro_s[i]);
    }
    Matrix intermediate;
    la::MulInit(ro_s_inv, ro_U_trans, &intermediate);
    la::MulInit(ro_VT_trans, intermediate, A_inv);
  }

 public:
  
  void Compute() {

    // temporary variables for multiindex looping
    ArrayList<int> heads;
    Vector weighted_values;

    printf("\nStarting naive LPR...\n");
    fx_timer_start(NULL, "naive_lpr_compute");

    // initialization of temporary variables for computation...
    heads.Init(qset_.n_rows() + 1);
    weighted_values.Init(total_num_coeffs_);    

    // compute unnormalized sum for the numerator vector and the denominator
    // matrix
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	const double *r_col = rset_.GetColumnPtr(r);
	const double r_target = rset_targets_[r];
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double kernel_value = kernel_.EvalUnnormOnSq(dsqd);

	// multiindex looping
	for(index_t i = 0; i < qset_.n_rows(); i++) {
	  heads[i] = 0;
	}
	heads[qset_.n_rows()] = MAXINT;

	weighted_values[0] = 1.0;
	for(index_t k = 1, t = 1, tail = 1; k <= lpr_order_; k++, tail = t) {
	  for(index_t i = 0; i < qset_.n_rows(); i++) {
	    int head = (int) heads[i];
	    heads[i] = t;
	    for(index_t j = head; j < tail; j++, t++) {

	      // compute numerator vector position t based on position j
	      weighted_values[t] = weighted_values[j] * r_col[i];
	    }
	  }
	}

	// tally up the sum here
	for(index_t i = 0; i < total_num_coeffs_; i++) {
	  numerator_[q][i] = numerator_[q][i] + r_target *
	    weighted_values[i] * kernel_value;

	  for(index_t j = 0; j < total_num_coeffs_; j++) {
	    denominator_[q].set(i, j, denominator_[q].get(i, j) + 
				weighted_values[i] * weighted_values[j] *
				kernel_value);
	  }
	}
      } // end of looping over each reference point
    } // end of looping over each query point

    // now iterate over all query points and compute regression estimate
    for(index_t q = 0; q < qset_.n_cols(); q++) {

      const double *q_col = qset_.GetColumnPtr(q);
      Matrix denominator_inv_q;
      Vector beta_q;

      // now invert the denominator matrix for each query point and multiply
      // by the numerator vector
      PseudoInverse(denominator_[q], &denominator_inv_q);      
      la::MulInit(denominator_inv_q, numerator_[q], &beta_q);

      // compute the vector [1, x_1, \cdots, x_D, second order, ...]
      weighted_values[0] = 1.0;
      for(index_t k = 1, t = 1, tail = 1; k <= lpr_order_; k++, tail = t) {
	for(index_t i = 0; i < qset_.n_rows(); i++) {
	  int head = (int) heads[i];
	  heads[i] = t;
	  for(index_t j = head; j < tail; j++, t++) {

	    // compute numerator vector position t based on position j
	    weighted_values[t] = weighted_values[j] * q_col[i];
	  }
	}
      }

      // compute the dot product between the multiindex vector for the query
      // point by the beta_q
      regression_values_[q] = la::Dot(weighted_values, beta_q);
    }
    
    fx_timer_stop(NULL, "naive_lpr_compute");
    printf("\nNaive LPR completed...\n");
  }

  void ReInit(int order) {

    // compute total number of coefficients
    total_num_coeffs_ = (int) math::BinomialCoefficient(order + qset_.n_rows(),
							qset_.n_rows());    
    regression_values_.SetZero();

    // reinitialize the temporary stroages for storing the numerator vectors
    // and the denominator matrices
    if(lpr_order_ != order) {
      lpr_order_ = order;
      total_num_coeffs_ = (int) 
	math::BinomialCoefficient(order + qset_.n_rows(), qset_.n_rows());
      
      numerator_.Destruct();
      numerator_.Init(qset_.n_cols());

      for(index_t i = 0; i < qset_.n_cols(); i++) {
	numerator_[i].Destruct();
	numerator_[i].Init(total_num_coeffs_, total_num_coeffs_);
	numerator_[i].SetZero();
	denominator_[i].Destruct();
	denominator_[i].Init(total_num_coeffs_, total_num_coeffs_);
	denominator_[i].SetZero();
      }
    }
  }

  void Init(int order) {

    // get datasets
    Dataset ref_dataset;

    // read the datasets
    const char *rfname = fx_param_str_req(NULL, "data");
    const char *qfname = fx_param_str(NULL, "query", rfname);

    // read reference dataset
    ref_dataset.InitFromFile(rfname);
    rset_.Own(&(ref_dataset.matrix()));

    // read the reference weights
    char *rtfname = NULL;
    if(fx_param_exists(NULL, "dtarget")) {
      rtfname = (char *)fx_param_str(NULL, "dtarget", NULL);
    }

    if(rtfname != NULL) {
      Dataset ref_targets;
      ref_targets.InitFromFile(rtfname);
      rset_targets_.Copy(ref_targets.matrix().GetColumnPtr(0),
			 ref_targets.matrix().n_rows());
    }
    else {
      rset_targets_.Init(rset_.n_cols());
      rset_targets_.SetAll(1);
    }

    if(!strcmp(qfname, rfname)) {
      qset_.Alias(rset_);
    }
    else {
      Dataset query_dataset;
      query_dataset.InitFromFile(qfname);
      qset_.Own(&(query_dataset.matrix()));
    }

    // get bandwidth
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));

    // compute total number of coefficients
    lpr_order_ = order;
    total_num_coeffs_ = (int) math::BinomialCoefficient(order + qset_.n_rows(),
							qset_.n_rows());
    
    // allocate temporary storages for storing the numerator vectors and
    // the denominator matrices
    numerator_.Init(qset_.n_cols());
    denominator_.Init(qset_.n_cols());

    for(index_t i = 0; i < qset_.n_cols(); i++) {
      numerator_[i].Init(total_num_coeffs_);
      numerator_[i].SetZero();
      denominator_[i].Init(total_num_coeffs_, total_num_coeffs_);
      denominator_[i].SetZero();
    }
    
    // allocate density storage
    regression_values_.Init(qset_.n_cols());
    regression_values_.SetZero();
  }

  void Init(Matrix &qset, Matrix &rset, int order) {

    // get datasets
    qset_.Alias(qset);
    rset_.Alias(rset);

    // get bandwidth
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));

    // compute total number of coefficients
    lpr_order_ = order;
    total_num_coeffs_ = (int) math::BinomialCoefficient(order + qset_.n_rows(),
							qset_.n_rows());
    
    // allocate temporary storages for storing the numerator vectors and
    // the denominator matrices
    numerator_.Init(qset_.n_cols(), total_num_coeffs_);
    denominator_.Init(qset_.n_cols());

    for(index_t i = 0; i < qset_.n_cols(); i++) {
      denominator_[i].Init(total_num_coeffs_, total_num_coeffs_);
      denominator_[i].SetZero();
    }
    
    // allocate density storage
    regression_values_.Init(qset.n_cols());
    regression_values_.SetZero();
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(NULL, "naive_lpr_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", regression_values_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }    
  }

  void ComputeMaximumRelativeError(const Vector &regression_estimate) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < regression_values_.length(); q++) {
      double rel_err = fabs(regression_estimate[q] - regression_values_[q]) / 
	regression_values_[q];
      
      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    fx_format_result(NULL, "maxium_relative_error_for_fast_LPR", "%g", 
		     max_rel_err);
  }

};


template<typename TKernel, typename TKernelAux>
class FastLpr {

  public:

  // forward declaration of LprStat class
  class LprStat;

  // our tree type using the KdeStat
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, LprStat > Tree;
  
  class LprStat {
  public:
    
    /** lower bound on the densities for the query points owned by this node 
     */
    ArrayList<double> mass_l_;
    
    /**
     * additional offset for the lower bound on the densities for the query
     * points owned by this node (for leaf nodes only).
     */
    ArrayList<double> more_l_;
    
    /**
     * lower bound offset passed from above
     */
    ArrayList<double> owed_l_;
    
    /** stores the portion pruned by finite difference
     */
    ArrayList<double> mass_e_;

    /** upper bound on the densities for the query points owned by this node 
     */
    ArrayList<double> mass_u_;
    
    /**
     * additional offset for the upper bound on the densities for the query
     * points owned by this node (for leaf nodes only)
     */
    ArrayList<double> more_u_;
    
    /**
     * upper bound offset passed from above
     */
    ArrayList<double> owed_u_;
    
    /** extra error that can be used for the query points in this node */
    ArrayList<double> mass_t_;
    
    /** Initialize the statistics */
    void Init() {
      mass_l_.Init();
      more_l_.Init();
      owed_l_.Init();
      mass_e_.Init();
      mass_u_.Init();
      more_u_.Init();
      owed_u_.Init();
      mass_t_.Init(); 
    }
    
    void Init(double bandwidth, 
	      typename TKernelAux::TSeriesExpansionAux *sea) {

    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
      Init();
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const LprStat& left_stat,
	      const LprStat& right_stat) {
      Init();
    }
    
    void Init(double bandwidth, const Vector& center,
	      typename TKernelAux::TSeriesExpansionAux *sea) {
      Init();
    }

    void MergeChildBounds(LprStat &left_stat, LprStat &right_stat) {

      /*
      // steal left and right children's tokens
      double min_mass_t = min(left_stat.mass_t_, right_stat.mass_t_);

      // improve lower and upper bound
      mass_l_ = max(mass_l_, min(left_stat.mass_l_, right_stat.mass_l_));
      mass_u_ = min(mass_u_, max(left_stat.mass_u_, right_stat.mass_u_));
      mass_t_ += min_mass_t;
      left_stat.mass_t_ -= min_mass_t;
      right_stat.mass_t_ -= min_mass_t;
      */
    }

    void PushDownTokens
      (LprStat &left_stat, LprStat &right_stat, double *de,
       typename TKernelAux::TLocalExpansion *local_expansion, double *dt) {
      
      /*
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
      */

    }

    LprStat() { }
    
    ~LprStat() {}
    
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
  Matrix rset_weights_;

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

  /** local polynomial approximation order */
  int lpr_order_;

  /** total number of coefficients for the local polynomial */
  int total_num_coeffs_;

  // member functions
  void UpdateBounds(Tree *qnode, Tree *rnode, 
		    double *dl, double *de, double *du, double *dt,
		    int *order_farfield_to_local, int *order_farfield,
		    int *order_local) {
  }

  /** exhaustive base KDE case */
  void FLprBase(Tree *qnode, Tree *rnode) {

  }

  /** checking for prunability of the query and the reference pair */
  int Prunable(Tree *qnode, Tree *rnode, DRange &dsqd_range,
	       DRange &kernel_value_range, double &dl, double &de,
	       double &du, double &dt) {

    // query node stat
    LprStat &stat = qnode->stat();
    
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

  /** canonical fast LPR case */
  void FLpr(Tree *qnode, Tree *rnode) {

    /** temporary variable for storing lower bound change */
    double dl = 0, de = 0, du = 0, dt = 0;
    int order_farfield_to_local = -1, order_farfield = -1, order_local = -1;

    // temporary variable for holding distance/kernel value bounds
    DRange dsqd_range;
    DRange kernel_value_range;

    // query node statistics
    LprStat &stat = qnode->stat();

    // left child and right child of query node statistics
    LprStat *left_stat = NULL;
    LprStat *right_stat = NULL;

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
    
    // for leaf query node
    if(qnode->is_leaf()) {
      
      // for leaf pairs, go exhaustive
      if(rnode->is_leaf()) {
	FLprBase(qnode, rnode);
	return;
      }
      
      // for non-leaf reference, expand reference node
      else {
	Tree *rnode_first = NULL, *rnode_second = NULL;
	BestNodePartners(qnode, rnode->left(), rnode->right(), &rnode_first,
			 &rnode_second);
	FLpr(qnode, rnode_first);
	FLpr(qnode, rnode_second);
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
	FLpr(qnode_first, rnode);
	FLpr(qnode_second, rnode);
	return;
      }

      // for non-leaf reference node, expand both query and reference nodes
      else {
	Tree *rnode_first = NULL, *rnode_second = NULL;
	stat.PushDownTokens(*left_stat, *right_stat, NULL, NULL, 
			    &stat.mass_t_);
	
	BestNodePartners(qnode->left(), rnode->left(), rnode->right(),
			 &rnode_first, &rnode_second);
	FLpr(qnode->left(), rnode_first);
	FLpr(qnode->left(), rnode_second);

	BestNodePartners(qnode->right(), rnode->left(), rnode->right(),
			 &rnode_first, &rnode_second);
	FLpr(qnode->right(), rnode_first);
	FLpr(qnode->right(), rnode_second);
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
  }

  /** post processing step */
  void PostProcess(Tree *qnode) {
    
    LprStat &stat = qnode->stat();

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

  void BuildReferenceWeights(const Vector &rset_targets) {
    rset_weights_.Init(rset_.num_cols(), );
  }

  public:

  // constructor/destructor
  FastLpr() {}

  ~FastLpr() { 
    
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
  }

  void Init(int order) {
    
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

    // initialize the local polynomial order
    lpr_order_ = order;
    total_num_coeffs_ = (int) math::BinomialCoefficient(order + qset_.n_rows(),
							qset_.n_rows());

    // initialize the kernel
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));

    // initialize the series expansion object
    if(qset_.n_rows() <= 2) {
      sea_.Init(fx_param_int(NULL, "order", 7), qset_.n_rows());
    }
    else if(qset_.n_rows() <= 3) {
      sea_.Init(fx_param_int(NULL, "order", 5), qset_.n_rows());
    }
    else if(qset_.n_rows() <= 5) {
      sea_.Init(fx_param_int(NULL, "order", 3), qset_.n_rows());
    }
    else if(qset_.n_rows() <= 6) {
      sea_.Init(fx_param_int(NULL, "order", 1), qset_.n_rows());
    }
    else {
      sea_.Init(fx_param_int(NULL, "order", 0), qset_.n_rows());
    }
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(NULL, "fast_lpr_output", NULL)) != NULL) {
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
