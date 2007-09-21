#ifndef MULTIBODY_H
#define MULTIBODY_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/series_expansion_aux.h"


template<typename TKernel, typename TKernelDerivative>
class MultitreeMultibody {

  FORBID_COPY(MultitreeMultibody);

public:

  /** Statatistics stored in each node of the tree */
  class MultibodyStat {
    
  public:
    /** lower index of the depth first order */
    int lo_index;
    
    /** high index of the depth first order */
    int hi_index;
    
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

    /** Initialize the statistics */
    void Init() {

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
  };

  typedef BinarySpaceTree<DHrectBound<2>, Matrix, MultibodyStat> Tree;
  
  MultitreeMultibody() {}
  
  ~MultitreeMultibody() { delete root_; }

  /** Main computation */
  void Compute() {
    ArrayList<Tree *> root_nodes;
    double total_n_tuples;

    // Warning, I should fix this so that it generalizes to any number of
    // tuples... May involves coming up with a solution that involves
    // templates...
    root_nodes.Init(3);
    tmp_nodes_.Init(3);
    non_leaf_indices_.Init(0);
    tmp_non_leaf_indices_.Init(0);
    leaf_indices_.Init(0);
    tmp_leaf_indices_.Init(0);
    
    // store node pointers
    for(index_t i = 0; i < 3; i++) {
      root_nodes[i] = root_;
    }
    
    // determine which are leaves and non-leaves...
    for(index_t i = 0; i < 3; i++) {
      if((root_nodes[i])->is_leaf()) {
	leaf_indices_.AddBackItem(i);
      }
      else {
	non_leaf_indices_.AddBackItem(i);
      }
    }
    
    total_n_tuples = ttn(0, root_nodes);
    MTMultibody(root_nodes, total_n_tuples);
  }

  /** Initialize the tree */
  void Init(Matrix& data) {
    fx_timer_start(NULL, "tree_d");
    tree::LoadKdTree(NULL, &data, &root_, NULL);
    fx_timer_stop(NULL, "tree_d");
  }

private:
  
  /** Temporary storage space for holding onto the node pointers */
  ArrayList<Tree *> tmp_nodes_;
  
  /** The current list of non-leaf indices */
  ArrayList<int> non_leaf_indices_;
  
  /** Storage for holding onto temporary non-leaf indices */
  ArrayList<int> tmp_non_leaf_indices_;

  /** The current list of leaf_indices */
  ArrayList<int> leaf_indices_;
  
  /** Storage for holding onto temporary leaf indices */
  ArrayList<int> tmp_leaf_indices_;

  /** pointer to the root of the tree */
  Tree *root_;

  bool as_indexes_strictly_surround_bs(Tree *a, Tree *b) {
    return (a->stat().lo_index < b->stat().lo_index &&
	    a->stat().hi_index >= b->stat().hi_index) ||
      (a->stat().lo_index <= b->stat().lo_index && 
       a->stat().hi_index > b->stat().hi_index);
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
    
    if ( b == n-1 ) {
      result = (double) bkn->count();
    }
    else {
      int j;
      bool conflict = 0;
      bool simple_product = 1;
      
      result = (double) bkn->count();
      
      for(j = b+1 ; j < n && !conflict; j++) {
	Tree *knj = nodes[j];
	if (bkn->stat().lo_index >= knj->stat().hi_index) {
	  conflict = 1;
	}
	else if(nodes[j-1]->stat().hi_index > knj->stat().lo_index) {
	  simple_product = 0;
	}
      }
      
      if ( conflict ) {
	result = 0.0;
      }
      else if ( simple_product ) {
	for ( j = b+1 ; j < n ; j++ ) {
	  result *= nodes[j]->count();
	}
      }
      else {
	bool jdiff = -1; 
	// undefined... will eventually point to the
	// lowest j > b such that nodes[j] is different from
	// bkn
	
	for ( j = b+1 ; jdiff < 0 && j < n ; j++ ) {
	  Tree *knj = nodes[j];
	  if(bkn->stat().lo_index != knj->stat().lo_index ||
	     bkn->stat().hi_index != knj->stat().hi_index) {
	    jdiff = j;
	  }
	}
	
	if(jdiff < 0) {
	  result = math::BinomialCoefficient(bkn->count(), n - b);
	}
	else {
	  Tree *dkn = nodes[jdiff];
	  
	  if(dkn->stat().lo_index >= bkn->stat().hi_index) {
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

    int max_num_points = 0;
    int max_index = -1;

    for(index_t i = 0; i < tmp_non_leaf_indices_.size(); i++) {

      int non_leaf_index = tmp_non_leaf_indices_[i];
      
      if(max_num_points < (nodes[non_leaf_index])->count()) {
	max_num_points = (nodes[non_leaf_index])->count();
	max_index = non_leaf_index;
      }
    }
    return max_index;
  }

  /** Pruning rule */
  int Prunable(ArrayList<Tree *> nodes) {
    return 0;
  }

  /** Base exhaustive case */
  void MTMultibodyBase(ArrayList<Tree *> nodes) {
    
  }

  /** Main multitree recursion */
  void MTMultibody(ArrayList<Tree *> nodes, double num_tuples) {
    
    if(Prunable(nodes)) {
      
    }

    // all leaves, then base case
    else if(non_leaf_indices_.size() == 0) {
      MTMultibodyBase(nodes);
    }
    
    // else, split an internal node and recurse
    else {
      int split_index;
      tmp_non_leaf_indices_.Resize(0);
      tmp_leaf_indices_.Resize(0);
      
      // save node pointers before recursing
      for(index_t i = 0; i < nodes.size(); i++) {
	tmp_nodes_[i] = nodes[i];
      }
      for(index_t i = 0; i < non_leaf_indices_.size(); i++) {
	tmp_non_leaf_indices_.AddBackItem(non_leaf_indices_[i]);
      }
      for(index_t i = 0; i < leaf_indices_.size(); i++) {
	tmp_leaf_indices_.AddBackItem(leaf_indices_[i]);
      }
      
      // apply splitting heuristic
      split_index = FindSplitNode(nodes);

      nodes[split_index] = tmp_nodes_[split_index]->left();
      
    }
  }

};

#endif
