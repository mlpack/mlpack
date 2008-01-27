#ifndef MULTIBODY_H
#define MULTIBODY_H

#include <values.h>

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/series_expansion_aux.h"
#include "multibody_kernel.h"

template<typename TMultibodyKernel>
class NaiveMultibody {

  FORBID_COPY(NaiveMultibody);

 private:

  /** Temporary space for storing indices selected for exhaustive computation
   */
  ArrayList<int> exhaustive_indices_;

  /** dataset for the tree */
  Matrix data_;

  /** multibody kernel function */
  TMultibodyKernel mkernel_;

  /** potential estimate */
  double potential_e_;

  /** exhaustive computer */
  void NMultibody(int level) {
    
    int num_nodes = mkernel_.order();
    int start_index = 0;
    double result = 0;

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
      result = mkernel_.Eval(data_, exhaustive_indices_);
      potential_e_ += result;
    }
  }

 public:

  NaiveMultibody() {}
  
  ~NaiveMultibody() {}

  void Init(const Matrix &data, double bandwidth) {
    data_.Alias(data);
    exhaustive_indices_.Init(3);
    mkernel_.Init(bandwidth);
    potential_e_ = 0;
  }

  void Compute() {

    NMultibody(0);

    printf("Got potential sum %g\n", potential_e_);
  }

};

template<typename TKernel, typename TKernelDerivative>
class MultibodyStat {

 public:


  /**
   * Far field expansion created by the reference points in this node.
   */
  FarFieldExpansion<TKernel, TKernelDerivative> farfield_expansion_;

  /**
   * Local expansion stored in this node.
   */
  LocalExpansion<TKernel, TKernelDerivative> local_expansion_;

  // getters and setters
  FarFieldExpansion<TKernel, TKernelDerivative> &get_farfield_coeffs() {
    return farfield_expansion_;
  }

  /** Initialize the statistics */
  void Init() {
  }

  void Init(double bandwidth, SeriesExpansionAux *sea) {
    farfield_expansion_.Init(bandwidth, sea);
    local_expansion_.Init(bandwidth, sea);
  }

  void Init(const Matrix& dataset, index_t &start, index_t &count) {
    Init();
  }

  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const MultibodyStat& left_stat,
	    const MultibodyStat& right_stat) {
    Init();
  }

  void Init(double bandwidth, const Vector& center,
	    SeriesExpansionAux *sea) {

    farfield_expansion_.Init(bandwidth, center, sea);
    local_expansion_.Init(bandwidth, center, sea);
  }

  MultibodyStat() { }

  ~MultibodyStat() {}

};


template<typename TMultibodyKernel, typename TKernel, 
	 typename TKernelDerivative>
class MultitreeMultibody {

  FORBID_COPY(MultitreeMultibody);

public:

  typedef BinarySpaceTree<DHrectBound<2>, Matrix, 
    MultibodyStat<TKernel, TKernelDerivative> > Tree;
  
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
    NumTupleNodesExpanded_ = 0;

    // first compute exhaustively for self-tuples
    printf("Starting with %g tuples...\n", total_num_tuples_);
    for(index_t i = 0; i < list_of_leaves_.size(); i++) {
      ArrayList<Tree *> base_nodes;

      base_nodes.Init(mkernel_.order());
      for(index_t k = 0; k < mkernel_.order(); k++) {
	base_nodes[k] = list_of_leaves_[i];
      }
      base_num_tuples_[base_indices_count_] = ttn(0, base_nodes);

      MTMultibodyBase(base_nodes, 0);
      total_num_tuples_ -= base_num_tuples_[i];
      base_indices_count_++;
    }
    printf("Accounted for %g tuples before the start out of %g tuples!\n", extra_token_,
	   total_num_tuples_);
    printf("Current lower bound %g\n", potential_l_);
    MTMultibody(root_nodes, total_num_tuples_);

    printf("Total potential estimate: %g\n", potential_e_);
    printf("Number of series approximations: %d\n", NumPrunes_);
    printf("Number of tuples of nodes expanded: %d\n", NumTupleNodesExpanded_);
  }

  void InitExpansionObjects(Tree *node) {
    
    if(node != NULL) {
      node->stat().Init(sqrt(mkernel_.bandwidth_sq()), &sea_);
      node->bound().CalculateMidpoint
	(&(node->stat().farfield_expansion_.get_center()));
      node->bound().CalculateMidpoint
	(&(node->stat().local_expansion_.get_center()));
    }

    if(!node->is_leaf()) {
      InitExpansionObjects(node->left());
      InitExpansionObjects(node->right());
    }
    else {
      list_of_leaves_.AddBackItem(node);
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
    sea_.Init(8, data_.n_rows());

    // initialize the multibody kernel and the series expansion objects
    // for all nodes
    mkernel_.Init(bandwidth);
    list_of_leaves_.Init(0);
    InitExpansionObjects(root_);
    printf("Got %d leaves...\n", list_of_leaves_.size());

    fx_timer_stop(NULL, "tree_d");

    // more temporary variables initialization
    non_leaf_indices_.Init(mkernel_.order());
    distmat_.Init(mkernel_.order(), mkernel_.order());
    exhaustive_indices_.Init(mkernel_.order());
    node_bounds_.Init(mkernel_.order());

    base_indices_count_ = 0;
    base_num_tuples_.Init(list_of_leaves_.size());
    for(index_t i = 0; i < base_num_tuples_.size(); i++) {
      base_num_tuples_[i] = 0;
    }

    // potential bounds and tokens initialized to 0.
    potential_l_ = potential_e_ = 0;
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
  SeriesExpansionAux sea_;
  
  /** multibody kernel function */
  MultibodyKernel mkernel_;

  /** the total number of n-tuples to consider */
  double total_num_tuples_;
  
  /** Extra amount of error that can be spent */
  double extra_token_;

  /** potential estimate */
  double potential_e_;
  
  /** Running lower bound on the potential */
  double potential_l_;
  
  /** approximation relative error bound */
  double tau_;

  /** list of leaf nodes */
  ArrayList<Tree *> list_of_leaves_;

  /** number of prunes made */
  int NumPrunes_;

  /** number of tuples of nodes expanded */
  int NumTupleNodesExpanded_;
  
  int base_indices_count_;

  /** the number of tuples for leaf node index (i, i, ... i) */
  ArrayList<double> base_num_tuples_;

  // functions

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
  
  /** get effective count of the node */
  double wrapper_ttn(ArrayList<Tree *> nodes) {

    double unfiltered_count = ttn(0, nodes);

    // depending on how many leaf tuples have been exhaustively computed
    // under the first node, take out the number of tuples

    for(index_t i = 0; i < list_of_leaves_.size(); i++) {
      int flag = 1;
      
      for(index_t j = 0; j < nodes.size(); j++) {
	
	if(!(nodes[j]->begin() <= list_of_leaves_[i]->begin() &&
	     list_of_leaves_[i]->end() <= nodes[j]->end())) {
	  flag = 0;
	  break;
	}
      }

      if(flag == 1) {
	unfiltered_count -= base_num_tuples_[i];
      }
    }

    return unfiltered_count;
  }

  /** Compute the total number of n-tuples */
  double ttn(int b, ArrayList<Tree *> nodes) {
    
    Tree *bkn = nodes[b];
    double result;
    int n = nodes.size();

    if(b == n - 1) {
      result = bkn->count();
    }
    else {
      int j;
      int conflict = 0;
      int simple_product = 1;
      
      result = bkn->count();
      
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
	    result = math::BinomialCoefficient(bkn->count(), 
					       jdiff - b);
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

      //int non_leaf_index = non_leaf_indices_[i];
      //double minimum_side_length = MAXDOUBLE;
      
      //find out the minimum side length
      //for(index_t j = 0; j < data_.n_rows(); j++) {
	
      //DRange range = nodes[non_leaf_index]->bound().get(j);
      //double side_length = range.width();
      
      //if(side_length < minimum_side_length) {
      //minimum_side_length = side_length;
      //	}
      //      }
      //      if(minimum_side_length > global_min) {
      //	global_min = minimum_side_length;
      //	global_index = non_leaf_index;
      //}

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

    double min_potential, max_potential;
    double lower_change;
    double error, estimate;

    // compute pairwise bounding box distances
    for(index_t i = 0; i < mkernel_.order(); i++) {
      node_bounds_[i] = &(nodes[i]->bound());
    }
    mkernel_.EvalNodes(node_bounds_, &min_potential, &max_potential);

    lower_change = num_tuples * min_potential;
    
    error = 0.5 * num_tuples * (max_potential - min_potential);
    
    estimate = 0.5 * num_tuples * (min_potential + max_potential);

    // compute whether the error is below the threshold
    *allowed_err = tau_ * (potential_l_ + lower_change) *
      ((num_tuples + extra_token_) / total_num_tuples_);

    if(likely(error >= 0) && 
       error <= tau_ * (potential_l_ + lower_change) *
       ((num_tuples + extra_token_) / total_num_tuples_)) {
      
      potential_l_ += lower_change;
      potential_e_ += estimate;

      extra_token_ = num_tuples + extra_token_ - 
	error * total_num_tuples_ / (tau_ * potential_l_);

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
      
      FarFieldExpansion<TKernel, TKernelDerivative> &coeffs0 =
	nodes[0]->stat().get_farfield_coeffs();
      FarFieldExpansion<TKernel, TKernelDerivative> &coeffs1 =
	nodes[1]->stat().get_farfield_coeffs();
      FarFieldExpansion<TKernel, TKernelDerivative> &coeffs2 =
	nodes[2]->stat().get_farfield_coeffs();
      double total_relerr = allowed_err / 
	(num_tuples * max_ij * max_ik * max_jk);
      double rel_err = max(pow(total_relerr + 1, 1.0 / 3.0) - 1, 0);
      
      // compute the required number of terms
      int order_ij = coeffs0.OrderForConvertingtoLocal(nodes[0]->bound(),
						       nodes[1]->bound(),
						       distmat.get(0, 1),
						       min_ij * rel_err,
						       &actual_error1);
      int order_ik = coeffs1.OrderForConvertingtoLocal(nodes[0]->bound(),
						       nodes[2]->bound(),
						       distmat.get(0, 2),
						       min_ik * rel_err,
						       &actual_error2);
      int order_jk = coeffs2.OrderForConvertingtoLocal(nodes[1]->bound(),
						       nodes[2]->bound(),
						       distmat.get(1, 2),
						       min_jk * rel_err,
						       &actual_error3);
      
      int max_order = coeffs0.get_max_order() / 2 - 1;
      if(order_ij >= 0 && order_ik >= 0 && order_jk >= 0 &&
	 order_ij < max_order && order_ik < max_order && 
	 order_jk < max_order &&
	 sea_.get_total_num_coeffs(order_ij) *
	 sea_.get_total_num_coeffs(order_ik) *
	 sea_.get_total_num_coeffs(order_jk) <
	 nodes[0]->count() * nodes[1]->count() * nodes[2]->count()) {

	coeffs0.RefineCoeffs(data_, weights_, nodes[0]->begin(), 
			     nodes[0]->end(), order_ij);
	coeffs1.RefineCoeffs(data_, weights_, nodes[1]->begin(), 
			     nodes[1]->end(), order_ik);
	coeffs2.RefineCoeffs(data_, weights_, nodes[2]->begin(), 
			     nodes[2]->end(), order_jk);
	
	potential_l_ += num_tuples * min_ij * min_ik * min_jk;
	potential_e_ += coeffs0.ConvolveField(coeffs1, coeffs2, order_ij, 
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
	  max_rel_err_incurred2 + max_rel_err_incurred3;
	double error = max_rel_err_incurred * max_ij * max_ik * max_jk;
	
	extra_token_ = num_tuples + extra_token_ - error * total_num_tuples_ /
	  (tau_ * potential_l_);
	
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
    double result;
    int num_nodes = nodes.size();

    if(level < num_nodes) {
      
      // run over each point in this node
      if(level > 0) {
	if(nodes[level - 1]->begin() == nodes[level]->begin()) {
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

      // complete the table of distance computation
      result = mkernel_.Eval(data_, exhaustive_indices_);

      potential_e_ += result;
      potential_l_ += result;	
    }
  }

  /** Main multitree recursion */
  void MTMultibody(ArrayList<Tree *> nodes, double num_tuples) {
    
    double allowed_err = 0;
    NumTupleNodesExpanded_++;

    if(Prunable(nodes, num_tuples, &allowed_err)) {
      return;
    }

    // figure out which ones are non-leaves
    non_leaf_indices_.Resize(0);
    for(index_t i = 0; i < mkernel_.order(); i++) {
      if(!(nodes[i]->is_leaf())) {
	non_leaf_indices_.AddBackItem(i);
      }
    }
    
    // all leaves and not all of them are equal, then base case
    if(non_leaf_indices_.size() == 0) {
      if(!(nodes[0] == nodes[1] && nodes[1] == nodes[2] &&
	   nodes[0] == nodes[2])) {
	MTMultibodyBase(nodes, 0);
	extra_token_ += num_tuples;
      }
      return;
    }
    
    // else, split an internal node and recurse
    else {
      int split_index;
      double new_num_tuples;
      
      // copy to new nodes
      ArrayList<Tree *> new_nodes;
      new_nodes.Init(mkernel_.order());
      for(index_t i = 0; i < mkernel_.order(); i++) {
	new_nodes[i] = nodes[i];
      }
      
      // apply splitting heuristic
      split_index = FindSplitNode(nodes);
      
      // recurse to the left
      new_nodes[split_index] = nodes[split_index]->left();
      new_num_tuples = wrapper_ttn(new_nodes);
      
      if(new_num_tuples > 0) {
	MTMultibody(new_nodes, new_num_tuples);
      }
      
      // recurse to the right
      new_nodes[split_index] = nodes[split_index]->right();
      new_num_tuples = wrapper_ttn(new_nodes);
      
      if(new_num_tuples > 0) {
	MTMultibody(new_nodes, new_num_tuples);
      }
    }
  }

};

#endif
