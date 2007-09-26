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

  /** Summed up potential for query points in this node */
  double potential_;

  /**
   * Extra amount of error that can be spent for the query points in
   * this node.
   */
  double extra_token_;

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
    potential_ = 0;
    extra_token_ = 0;
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

    potential_ = 0;
    extra_token_ = 0;
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
  
  ~MultitreeMultibody() { delete root_; }

  // getters/setters
  
  const Matrix &get_data() const { return data_; }

  // interesting functions...

  /** Main computation */
  void Compute(double tau) {

    ArrayList<Tree *> root_nodes;
    root_nodes.Init(3);

    // store node pointers
    for(index_t i = 0; i < 3; i++) {
      root_nodes[i] = root_;
    }
    
    total_num_tuples_ = math::BinomialCoefficient(data_.n_cols(), 
						  mkernel_.order());
    tau_ = tau;

    // run and do timing for multitree multibody
    MTMultibody(root_nodes, total_num_tuples_);

    printf("Total potential estimate: %g\n", potential_e_);
  }

  void InitExpansionObjects(Tree *node) {
    
    if(node != NULL) {
      Vector far_center;
      Vector local_center;
      far_center.Alias(node->stat().farfield_expansion_.get_center());
      local_center.Alias(node->stat().local_expansion_.get_center());
      node->bound().CalculateMidpoint(&far_center);
      node->bound().CalculateMidpoint(&local_center); 
      node->stat().Init(sqrt(mkernel_.bandwidth_sq()), &sea_);
    }

    if(!node->is_leaf()) {
      InitExpansionObjects(node->left());
      InitExpansionObjects(node->right());
    }
  }

  /** Initialize the kernel object, and build the tree */
  void Init(double bandwidth) {

    fx_timer_start(NULL, "tree_d");
    tree::LoadKdTree(NULL, &data_, &root_, NULL);
    weights_.Init(data_.n_cols());
    
    // by default, each point has a uniform weight
    weights_.SetAll(1);

    sea_.Init(10, data_.n_rows());

    mkernel_.Init(bandwidth);
    InitExpansionObjects(root_);

    fx_timer_stop(NULL, "tree_d");

    non_leaf_indices_.Init(mkernel_.order());
    distmat_.Init(mkernel_.order(), mkernel_.order());
    exhaustive_indices_.Init(mkernel_.order());
    node_bounds_.Init(mkernel_.order());

    potential_l_ = potential_e_ = 0;
    extra_token_ = 0;
  }

private:
  
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
  
  /**
   * Extra amount of error that can be spent
   */
  double extra_token_;

  /** potential estimate */
  double potential_e_;
  
  /** Running lower bound on the potential */
  double potential_l_;

  /** approximation relative error bound */
  double tau_;

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

      extra_token_ = num_tuples + extra_token_ - error * total_num_tuples_ /
	(tau_ * potential_l_);

      DEBUG_ASSERT(extra_token_ >= 0);
      return 1;
    }
    return 0;
  }

  /** Pruning rule for series approxiamation approach */
  int PrunableSeriesExpansion(ArrayList<Tree *> nodes, double num_tuples,
			      double allowed_err) {
    
    Matrix distmat;
    distmat.Alias(mkernel_.EvalMinMaxDsqds(node_bounds_));
    
    //double max_ij = mkernel_.EvalUnnormOnSqOnePair(distmat.get(0, 1));
    //double max_ik = mkernel_.EvalUnnormOnSqOnePair(distmat.get(0, 2));
    //double max_jk = mkernel_.EvalUnnormOnSqOnePair(distmat.get(1, 2));
    double min_ij = mkernel_.EvalUnnormOnSqOnePair(distmat.get(1, 0));
    double min_ik = mkernel_.EvalUnnormOnSqOnePair(distmat.get(2, 0));
    double min_jk = mkernel_.EvalUnnormOnSqOnePair(distmat.get(2, 1));
    
    //double total_relerr = allowed_err / 
    //(num_tuples * max_ij * max_ik * max_jk);
    //double rel_err = max(pow(total_relerr + 1, 1.0 / 3.0) - 1, 0);

    // compute the required number of terms
    int order_ij = 10;
    int order_ik = 10;
    int order_jk = 10;
    
    nodes[0]->stat().get_farfield_coeffs().
      AccumulateCoeffs(data_, weights_, nodes[0]->begin(), nodes[0]->end(),
		       order_ij);
    nodes[1]->stat().get_farfield_coeffs().
      AccumulateCoeffs(data_, weights_, nodes[1]->begin(), nodes[1]->end(),
		       order_ik);
    nodes[2]->stat().get_farfield_coeffs().
      AccumulateCoeffs(data_, weights_, nodes[2]->begin(), nodes[2]->end(),
		       order_jk);
    
    potential_l_ += num_tuples * min_ij * min_ik * min_jk;
    potential_e_ += 
      (nodes[0]->stat().get_farfield_coeffs()).ConvolveField
      (nodes[1]->stat().get_farfield_coeffs(), 
       nodes[2]->stat().get_farfield_coeffs(), 2, 2, 2);
    
    return 0;
  }

  /** Base exhaustive case */
  void MTMultibodyBase(ArrayList<Tree *> nodes, int level) {

    int start_index;
    double result;
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

      // complete the table of distance computation
      result = mkernel_.Eval(data_, exhaustive_indices_);
      potential_e_ += result;
      potential_l_ += result;
    }
  }

  /** Main multitree recursion */
  void MTMultibody(ArrayList<Tree *> nodes, double num_tuples) {
    
    double allowed_err = 0;
    if(Prunable(nodes, num_tuples, &allowed_err)) {
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
