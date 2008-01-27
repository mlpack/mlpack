#ifndef MULTIBODY_H
#define MULTIBODY_H

#include <values.h>

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/series_expansion_aux.h"
#include "multibody_kernel.h"

template<typename TMultibodyKernel>
class NaiveMultibody {

  FORBID_ACCIDENTAL_COPIES(NaiveMultibody);

 private:

  /** Temporary space for storing indices selected for exhaustive computation
   */
  ArrayList<int> exhaustive_indices_;

  /** dataset for the tree */
  Matrix data_;

  /** multibody kernel function */
  TMultibodyKernel mkernel_;

  /** potential estimate */
  double neg_potential_e_;
  double pos_potential_e_;

  /** exhaustive computer */
  void NMultibody(int level) {
    
    int num_nodes = mkernel_.order();
    int start_index = 0;
    double neg, pos;

    if(level < num_nodes) {
      
      if(level == 0) {
	start_index = 0;
      }
      else {
	start_index = exhaustive_indices_[level - 1] + 1;
      }
      
      for(index_t i = start_index; i < data_.n_cols() - 
	    (num_nodes - level - 1); i++) {
	exhaustive_indices_[level] = i;
	NMultibody(level + 1);
      }
    }
    else {
      mkernel_.Eval(data_, exhaustive_indices_, &neg, &pos);
      neg_potential_e_ += neg;
      pos_potential_e_ += pos;
    }
  }

 public:

  NaiveMultibody() {}
  
  ~NaiveMultibody() {}

  void Init(const Matrix &data, double bandwidth) {
    data_.Alias(data);
    exhaustive_indices_.Init(3);
    mkernel_.Init(bandwidth);
    neg_potential_e_ = pos_potential_e_ = 0;
  }

  void Compute() {

    NMultibody(0);

    printf("Negative potential sum %g\n", neg_potential_e_);
    printf("Positive potential sum %g\n", pos_potential_e_);
    printf("Got potential sum %g\n", neg_potential_e_ + pos_potential_e_);
  }

};

template<typename TKernelAux>
class MultibodyStat {

 public:

  /**
   * Far field expansion created by the reference points in this node.
   */
  FarFieldExpansion<TKernelAux> farfield_expansion_;

  /**
   * Local expansion stored in this node.
   */
  LocalExpansion<TKernelAux> local_expansion_;

  // getters and setters
  FarFieldExpansion<TKernelAux> &get_farfield_coeffs() {
    return farfield_expansion_;
  }

  /** Initialize the statistics */
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
	    const MultibodyStat& left_stat,
	    const MultibodyStat& right_stat) {
    Init();
  }

  void Init(const Vector& center, const TKernelAux &ka) {
    farfield_expansion_.Init(center, ka);
    local_expansion_.Init(center, ka);
  }

  MultibodyStat() { }

  ~MultibodyStat() {}

};


template<typename TMultibodyKernel, typename TKernelAux>
class MultitreeMultibody {

  FORBID_ACCIDENTAL_COPIES(MultitreeMultibody);

public:

  typedef BinarySpaceTree<DHrectBound<2>, Matrix, 
    MultibodyStat<TKernelAux> > Tree;
  
  typedef TMultibodyKernel MultibodyKernel;

  // constructor/destructor
  MultitreeMultibody() {}
  
  ~MultitreeMultibody() { 
    delete root_;
  }

  // getters/setters
  
  const Matrix &get_data() const { return data_; }

  // interesting functions...

  /** Main computation */
  void Compute(double tau) {
    
    ArrayList<Tree *> root_nodes;
    root_nodes.Init(mkernel_.order());

    // store node pointers
    for(index_t i = 0; i < mkernel_.order(); i++) {
      root_nodes[i] = root_;
    }
    
    total_num_tuples_ = math::BinomialCoefficient(data_.n_cols(), 
						  mkernel_.order());
    tau_ = tau;

    // run and do timing for multitree multibody
    NumPrunes_ = 0;
    NumNodesExpanded_ = 0;
    MTMultibody(root_nodes, total_num_tuples_);

    printf("Negative potential %g\n", neg_potential_e_);
    printf("Positive potential %g\n", pos_potential_e_);
    printf("Total potential estimate: %g\n", pos_potential_e_ +
	   neg_potential_e_);
    printf("Number of series approximations: %d\n", NumPrunes_);
    printf("Number of tuples of nodes expanded: %d\n", NumNodesExpanded_);
  }

  void InitExpansionObjects(Tree *node) {
    
    if(node != NULL) {
      node->stat().Init(ka_);
      node->bound().CalculateMidpoint
	(node->stat().farfield_expansion_.get_center());
      node->bound().CalculateMidpoint
	(node->stat().local_expansion_.get_center());
    }

    if(!node->is_leaf()) {
      InitExpansionObjects(node->left());
      InitExpansionObjects(node->right());
    }
  }

  /** Initialize the kernel object, and build the tree */
  void Init(double bandwidth) {

    const char *fname = fx_param_str(NULL, "data", NULL);
    int leaflen = fx_param_int(NULL, "leaflen", 20);
     
    // read in the dataset and build a kd-tree
    fx_timer_start(NULL, "tree_d");
    Dataset dataset_;
    dataset_.InitFromFile(fname);
    data_.Own(&(dataset_.matrix()));
    root_ = tree::MakeKdTreeMidpoint<Tree>(data_, leaflen, NULL);
    weights_.Init(data_.n_cols());
    
    // by default, each point has a uniform weight
    weights_.SetAll(1);

    // set the maximum order of approximation here!
    ka_.Init(bandwidth, 4, data_.n_rows());

    // initialize the multibody kernel and the series expansion objects
    // for all nodes
    mkernel_.Init(bandwidth);
    InitExpansionObjects(root_);

    fx_timer_stop(NULL, "tree_d");

    // more temporary variables initialization
    non_leaf_indices_.Init(mkernel_.order());
    distmat_.Init(mkernel_.order(), mkernel_.order());
    exhaustive_indices_.Init(mkernel_.order());
    node_bounds_.Init(mkernel_.order());

    // initialize the combination generator
    combination_.Init(mkernel_.order());
    for(index_t i = 0; i < mkernel_.order(); i++) {
      combination_[i] = i;
    }
    combination_rank_ = 0;

    // potential bounds and tokens initialized to 0.
    neg_potential_u_ = neg_potential_e_ = 0;
    pos_potential_l_ = pos_potential_e_ = 0;
    extra_token_ = 0;
  }

private:

  // member variables

  /** hrect bounds passed to evaluation */
  ArrayList<DHrectBound<2> *> node_bounds_;

  /** The current list of non-leaf indices */
  ArrayList<int> non_leaf_indices_;

  /** Temporary space for storing indices selected for exhaustive computation 
   */
  ArrayList<int> exhaustive_indices_;

  /** Temporary space for storing pairwise distances */
  Matrix distmat_;

  /** pointer to the root of the tree */
  Tree *root_;

  /** dataset for the tree */
  Matrix data_;

  /** weight for each point */
  Vector weights_;

  /** series approximation auxiliary computations */
  TKernelAux ka_;
  
  /** multibody kernel function */
  MultibodyKernel mkernel_;

  /** the total number of n-tuples to consider */
  double total_num_tuples_;
  
  /** Extra amount of error that can be spent */
  double extra_token_;

  /** negative potential estimate */
  double neg_potential_e_;
  
  /** running lower bound on the negative potential */
  double neg_potential_u_;

  /** positive potential estimate */
  double pos_potential_e_;

  /** running lower bound on the positive potential */
  double pos_potential_l_;

  /** approximation relative error bound */
  double tau_;

  /** number of prunes made */
  int NumPrunes_;

  /** number of nodes expanded */
  int NumNodesExpanded_;
  
  /** index enumerating a combination from beginning to the end */
  Vector combination_;

  /** rank of the current combination */
  int combination_rank_;

  // functions

  /** combination enumerator */  
  success_t generate_next_symmetric_index(Vector &index) {

    int i, ok_so_far;
    int n = index.length();
    int top = n-1;
    
    do {
      index[top] += 1;
      ok_so_far = 1;
      
      if (index[top] >= data_.n_cols()) {
	index[top] = -1;
	top -= 1;
	ok_so_far = 0;
	
	if (top < 0) { 
	  return SUCCESS_FAIL;
	}
      }
      for (i = 0; i < top && ok_so_far; i++) {
	if (index[top] <= index[i]) {
	  ok_so_far = 0;
	}
      }
      if(ok_so_far) {
	top += 1;
      }
    }
    while (top < n);
    
    return SUCCESS_PASS;
  }

  /** test whether node a is an ancestor node of node b */
  int as_indexes_strictly_surround_bs(Tree *a, Tree *b) {
    return (a->begin() < b->begin() && a->end() >= b->end()) ||
      (a->begin() <= b->begin() && a->end() > b->end());
  }

  /** 
   * Compute the total number of n-tuples by recursively splitting up
   * the i-th node 
   */
  double two_ttn(int b, ArrayList<Tree *> nodes, int i) {

    double result = 0.0;
    Tree *kni = nodes[i];
    nodes[i] = kni->left();
    result += ttn(b, nodes);
    nodes[i] = kni->right();
    result += ttn(b, nodes);
    nodes[i] = kni;
    return result;
  }

  /** Compute the total number of n-tuples */
  double ttn(int b, ArrayList<Tree *> nodes) {
    
    Tree *bkn = nodes[b];
    double result;
    int n = nodes.size();

    if(b == n - 1) {
      result = (double) bkn->count();
    }
    else {
      int j;
      int conflict = 0;
      int simple_product = 1;
      
      result = (double) bkn->count();
      
      for(j = b + 1 ; j < n && !conflict; j++) {
	Tree *knj = nodes[j];
	
	if (bkn->begin() >= knj->end() - 1) {
	  conflict = 1;
	}
	else if(nodes[j - 1]->end() - 1 > knj->begin()) {
	  simple_product = 0;
	}
      }
      
      if(conflict) {
	result = 0.0;
      }
      else if(simple_product) {
	for(j = b + 1; j < n; j++) {
	  result *= nodes[j]->count();
	}
      }
      else {
	int jdiff = -1; 

	// undefined... will eventually point to the
	// lowest j > b such that nodes[j] is different from
	// bkn	
	for(j = b + 1; jdiff < 0 && j < n; j++) {
	  Tree *knj = nodes[j];
	  if(bkn->begin() != knj->begin() ||
	     bkn->end() - 1 != knj->end() - 1) {
	    jdiff = j;
	  }
	}

	if(jdiff < 0) {
	  result = math::BinomialCoefficient(bkn->count(), n - b);
	}
	else {
	  Tree *dkn = nodes[jdiff];

	  if(dkn->begin() >= bkn->end() - 1) {
	    result = math::BinomialCoefficient(bkn->count(), jdiff - b);
	    if(result > 0.0) {
	      result *= ttn(jdiff, nodes);
	    }
	  }
	  else if(as_indexes_strictly_surround_bs(bkn, dkn)) {
	    result = two_ttn(b, nodes, b);
	  }
	  else if(as_indexes_strictly_surround_bs(dkn, bkn)) {
	    result = two_ttn(b, nodes, jdiff);
	  }
	}
      }
    }
    return result;
  }

  /** Heuristic for node splitting - find the node with most points */
  int FindSplitNode(ArrayList<Tree *> nodes) {

    int global_index = -1;
    int global_min = 0;

    for(index_t i = 0; i < non_leaf_indices_.size(); i++) {

      /*
      int non_leaf_index = non_leaf_indices_[i];
      double minimum_side_length = MAXDOUBLE;

      // find out the minimum side length
      for(index_t j = 0; j < data_.n_rows(); j++) {
	
	DRange range = nodes[non_leaf_index]->bound().get(j);
	double side_length = range.width();
	
	if(side_length < minimum_side_length) {
	  minimum_side_length = side_length;
	}
      }
      if(minimum_side_length > global_min) {
	global_min = minimum_side_length;
	global_index = non_leaf_index;
      }
      */
      int non_leaf_index = non_leaf_indices_[i];
      if(nodes[non_leaf_index]->count() > global_min) {
	global_min = nodes[non_leaf_index]->count();
	global_index = non_leaf_index;
      }
    }
    return global_index;
  }

  /** Pruning rule */
  int Prunable(ArrayList<Tree *> nodes, double num_tuples, 
	       double *allowed_err) {

    double pos_min_potential, pos_max_potential;
    double neg_min_potential, neg_max_potential;
    double pos_lower_change, neg_upper_change;
    double pos_error, pos_estimate, neg_error, neg_estimate;
    double error;

    // compute pairwise bounding box distances
    for(index_t i = 0; i < mkernel_.order(); i++) {
      node_bounds_[i] = &(nodes[i]->bound());
    }
    mkernel_.EvalNodes(node_bounds_, &neg_min_potential, &neg_max_potential,
		       &pos_min_potential, &pos_max_potential);

    if(isnan(pos_max_potential) || isinf(pos_max_potential)) {
      return 0;
    }

    pos_lower_change = num_tuples * pos_min_potential;
    
    pos_error = 0.5 * num_tuples * (pos_max_potential - pos_min_potential);
    
    pos_estimate = 0.5 * num_tuples * (pos_min_potential + pos_max_potential);
    
    neg_upper_change = num_tuples * neg_max_potential;
    
    neg_error = 0.5 * num_tuples * (neg_max_potential - neg_min_potential);
    
    neg_estimate = 0.5 * num_tuples * (neg_min_potential + neg_max_potential);

    // compute whether the error is below the threshold
    *allowed_err = tau_ * (pos_potential_l_ + pos_lower_change -
			   (neg_potential_u_ + neg_upper_change)) *
      ((num_tuples + extra_token_) / total_num_tuples_);

    error = max(pos_error, neg_error);

    if(likely(error >= 0) && error <= (*allowed_err)) {
      
      pos_potential_l_ += pos_lower_change;
      pos_potential_e_ += pos_estimate;
      neg_potential_u_ += neg_upper_change;
      neg_potential_e_ += neg_estimate;

      extra_token_ = num_tuples + extra_token_ - error * total_num_tuples_ /
	(tau_ * (pos_potential_l_ -neg_potential_u_));
      
      DEBUG_ASSERT(extra_token_ >= 0);
      return 1;
    }

    return 0;
  }

  /** Pruning rule for series approxiamation approach */
  int PrunableSeriesExpansion(ArrayList<Tree *> nodes, double num_tuples,
			      double allowed_err) {

    if(nodes[0] != nodes[1] && nodes[0] != nodes[2] && nodes[1] != nodes[2]) {

      Matrix distmat;
      double actual_error1 = 0;
      double actual_error2 = 0;
      double actual_error3 = 0;
      distmat.Alias(mkernel_.EvalMinMaxDsqds(node_bounds_));
      
      double max_ij = mkernel_.EvalUnnormOnSqOnePair(distmat.get(0, 1));
      double max_ik = mkernel_.EvalUnnormOnSqOnePair(distmat.get(0, 2));
      double max_jk = mkernel_.EvalUnnormOnSqOnePair(distmat.get(1, 2));
      double min_ij = mkernel_.EvalUnnormOnSqOnePair(distmat.get(1, 0));
      double min_ik = mkernel_.EvalUnnormOnSqOnePair(distmat.get(2, 0));
      double min_jk = mkernel_.EvalUnnormOnSqOnePair(distmat.get(2, 1));
      
      FarFieldExpansion<TKernelAux> &coeffs0 =
	nodes[0]->stat().get_farfield_coeffs();
      FarFieldExpansion<TKernelAux> &coeffs1 =
	nodes[1]->stat().get_farfield_coeffs();
      FarFieldExpansion<TKernelAux> &coeffs2 =
	nodes[2]->stat().get_farfield_coeffs();
      double total_relerr = allowed_err / 
	(num_tuples * max_ij * max_ik * max_jk);
      double rel_err = max(pow(total_relerr + 1, 1.0 / 3.0) - 1, 0.0);
      
      // compute the required number of terms
      int order_ij = coeffs0.OrderForConvertingToLocal(nodes[0]->bound(),
						       nodes[1]->bound(),
						       distmat.get(0, 1),
						       distmat.get(1, 0),
						       min_ij * rel_err,
						       &actual_error1);
      int order_ik = coeffs1.OrderForConvertingToLocal(nodes[0]->bound(),
						       nodes[2]->bound(),
						       distmat.get(0, 2),
						       distmat.get(2, 0),
						       min_ik * rel_err,
						       &actual_error2);
      int order_jk = coeffs2.OrderForConvertingToLocal(nodes[1]->bound(),
						       nodes[2]->bound(),
						       distmat.get(1, 2),
						       distmat.get(2, 1),
						       min_jk * rel_err,
						       &actual_error3);
      
      int max_order = coeffs0.get_max_order() / 2 - 1;
      if(order_ij >= 0 && order_ik >= 0 && order_jk >= 0 &&
	 order_ij < max_order && order_ik < max_order && 
	 order_jk < max_order &&
	 ka_.sea_.get_total_num_coeffs(order_ij) *
	 ka_.sea_.get_total_num_coeffs(order_ik) *
	 ka_.sea_.get_total_num_coeffs(order_jk) <
	 nodes[0]->count() * nodes[1]->count() * nodes[2]->count()) {

	coeffs0.RefineCoeffs(data_, weights_, nodes[0]->begin(), 
			     nodes[0]->end(), order_ij);
	coeffs1.RefineCoeffs(data_, weights_, nodes[1]->begin(), 
			     nodes[1]->end(), order_ik);
	coeffs2.RefineCoeffs(data_, weights_, nodes[2]->begin(), 
			     nodes[2]->end(), order_jk);
	
	pos_potential_l_ += num_tuples * min_ij * min_ik * min_jk;
	pos_potential_e_ += coeffs0.ConvolveField(coeffs1, coeffs2, order_ij, 
						  order_ik, order_jk);

	// the maximum relative error incurred
	double max_rel_err_incurred1 = actual_error1 / min_ij;
	double max_rel_err_incurred2 = actual_error2 / min_ik;
	double max_rel_err_incurred3 = actual_error3 / min_jk;
	double max_rel_err_incurred =
	  max_rel_err_incurred1 + max_rel_err_incurred2 +
	  max_rel_err_incurred3 + max_rel_err_incurred1 *
	  max_rel_err_incurred2 + max_rel_err_incurred1 *
	  max_rel_err_incurred3 + max_rel_err_incurred2 *
	  max_rel_err_incurred3 + max_rel_err_incurred1 *
	  max_rel_err_incurred2 * max_rel_err_incurred3;
	double error = max_rel_err_incurred * max_ij * max_ik * max_jk;
	
	extra_token_ = num_tuples + extra_token_ - error * total_num_tuples_ /
	  (tau_ * pos_potential_l_);
	
	DEBUG_ASSERT(extra_token_ >= 0);
	return 1;
      }
      return 0;
    }
    
    return 0;
  }

  /** Pruning rule for series approxiamation second approach */
  int PrunableSeriesExpansion2(ArrayList<Tree *> nodes, double num_tuples,
			       double allowed_err) {

    if(nodes[0] != nodes[1] && nodes[0] != nodes[2] && nodes[1] != nodes[2]) {

      Matrix distmat;
      double actual_error2 = 0;
      double actual_error3 = 0;
      distmat.Alias(mkernel_.EvalMinMaxDsqds(node_bounds_));
            
      double max_ik = mkernel_.EvalUnnormOnSqOnePair(distmat.get(0, 2));
      double max_jk = mkernel_.EvalUnnormOnSqOnePair(distmat.get(1, 2));
      double min_ij = mkernel_.EvalUnnormOnSqOnePair(distmat.get(1, 0));
      double min_ik = mkernel_.EvalUnnormOnSqOnePair(distmat.get(2, 0));
      double min_jk = mkernel_.EvalUnnormOnSqOnePair(distmat.get(2, 1));
      
      FarFieldExpansion<TKernelAux> &coeffs0 =
	nodes[0]->stat().get_farfield_coeffs();
      FarFieldExpansion<TKernelAux> &coeffs1 =
	nodes[1]->stat().get_farfield_coeffs();
      FarFieldExpansion<TKernelAux> &coeffs2 =
	nodes[2]->stat().get_farfield_coeffs();
      double total_relerr = allowed_err / 
	(num_tuples * max_ik * max_jk);
      double rel_err = max(pow(total_relerr + 1, 1.0 / 2.0) - 1, 0.0);

      // compute the required number of terms
      int order_ik = -1;
      int order_jk = -1;
      if(min_ik * rel_err > 0 && min_jk * rel_err > 0) {
	order_ik = coeffs1.OrderForConvertingToLocal(nodes[0]->bound(),
						     nodes[2]->bound(),
						     distmat.get(0, 2),
						     distmat.get(2, 0),
						     min_ik * rel_err,
						     &actual_error2);
	order_jk = coeffs2.OrderForConvertingToLocal(nodes[1]->bound(),
						     nodes[2]->bound(),
						     distmat.get(1, 2),
						     distmat.get(2, 1),
						     min_jk * rel_err,
						     &actual_error3);
      }

      int max_order = coeffs1.get_max_order() / 2 - 1;
      if(order_ik >= 0 && order_jk >= 0 && order_ik < max_order && 
	 order_jk < max_order &&
	 ka_.sea_.get_total_num_coeffs(order_ik) *
	 ka_.sea_.get_total_num_coeffs(order_jk) < 2 * nodes[2]->count()) {

	coeffs2.RefineCoeffs(data_, weights_, nodes[2]->begin(), 
			     nodes[2]->end(), order_jk);
	
	pos_potential_l_ += num_tuples * min_ij * min_ik * min_jk;
	pos_potential_e_ += coeffs0.MixField(data_, nodes[0]->begin(),
					     nodes[0]->end(), 
					     nodes[1]->begin(),
					     nodes[1]->end(), coeffs1, 
					     coeffs2, order_ik, order_jk);

	// the maximum relative error incurred
	double max_rel_err_incurred2 = actual_error2 / min_ik;
	double max_rel_err_incurred3 = actual_error3 / min_jk;
	double max_rel_err_incurred =
	  max_rel_err_incurred2 + max_rel_err_incurred3 + 
	  max_rel_err_incurred2 * max_rel_err_incurred3;
	double error = max_rel_err_incurred * max_ik * max_jk;
	
	extra_token_ = num_tuples + extra_token_ - error * total_num_tuples_ /
	  (tau_ * pos_potential_l_);
	
	DEBUG_ASSERT(extra_token_ >= 0);
	return 1;
      }
      return 0;
    }
    return 0;
  }

  /** Base exhaustive case */
  void MTMultibodyBase(ArrayList<Tree *> nodes, int level) {

    int start_index;
    int num_nodes = nodes.size();

    if(level < num_nodes) {
      
      /* run over each point in this node */
      if(level > 0) {
	if(nodes[level - 1] == nodes[level]) {
	  start_index = exhaustive_indices_[level - 1] + 1;
	}
	else {
	  start_index = nodes[level]->begin();
	}
      }
      else {
	start_index = nodes[level]->begin();
      }
      
      for(index_t i = start_index; i < (nodes[level])->end(); i++) {	
	exhaustive_indices_[level] = i;
	MTMultibodyBase(nodes, level + 1);
      }
    }
    else {

      double neg, pos;

      // complete the table of distance computation
      mkernel_.Eval(data_, exhaustive_indices_, &neg, &pos);
      
      neg_potential_e_ += neg;
      neg_potential_u_ += neg;
      pos_potential_e_ += pos;
      pos_potential_l_ += pos;
    }
  }

  /** Main multitree recursion */
  void MTMultibody(ArrayList<Tree *> nodes, double num_tuples) {
    
    double allowed_err = 0;
    NumNodesExpanded_++;
    
    if(Prunable(nodes, num_tuples, &allowed_err)) {
      return;
    }
    else if(PrunableSeriesExpansion2(nodes, num_tuples, allowed_err)) {
      NumPrunes_++;
      return;
    }

    // figure out which ones are non-leaves
    non_leaf_indices_.Resize(0);
    for(index_t i = 0; i < 3; i++) {
      if(!(nodes[i]->is_leaf())) {
	non_leaf_indices_.AddBackItem(i);
      }
    }
    
    // all leaves, then base case
    if(non_leaf_indices_.size() == 0) {
      MTMultibodyBase(nodes, 0);
      extra_token_ += num_tuples;
      return;
    }
    
    // else, split an internal node and recurse
    else {
      int split_index;
      double new_num_tuples;
      
      // copy to new nodes
      ArrayList<Tree *> new_nodes;
      new_nodes.Init(3);
      for(index_t i = 0; i < 3; i++) {
	new_nodes[i] = nodes[i];
      }

      // apply splitting heuristic
      split_index = FindSplitNode(nodes);

      // recurse to the left
      new_nodes[split_index] = nodes[split_index]->left();
      new_num_tuples = ttn(0, new_nodes);
      
      if(new_num_tuples > 0) {
	MTMultibody(new_nodes, new_num_tuples);
      }
      
      // recurse to the right
      new_nodes[split_index] = nodes[split_index]->right();
      new_num_tuples = ttn(0, new_nodes);
      
      if(new_num_tuples > 0) {
	MTMultibody(new_nodes, new_num_tuples);
      }
    }
  }

};

#endif
