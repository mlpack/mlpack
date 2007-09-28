#ifndef ORTHO_RANGE_SEARCH_H
#define ORTHO_RANGE_SEARCH_H

#include <values.h>
#include "range_reader.h"
#include "fastlib/fastlib_int.h"

/** Naive orthogonal range search class */
class NaiveOrthoRangeSearch {
  
  FORBID_COPY(NaiveOrthoRangeSearch);

 private:

  /** tells whether the i-th point is in the specified orthogonal range */
  ArrayList<bool> in_range_;

  /** pointer to the dataset */
  Matrix data_;

  /** 
   * orthogonal range to search in: this will be generalized to a list
   * of orthogonal ranges
   */
  ArrayList<DRange> range_;
  
 public:

  NaiveOrthoRangeSearch() {}

  ~NaiveOrthoRangeSearch() {}

  /** get the result of the search */
  const ArrayList<bool>& get_results() const {
    return in_range_;
  }

  /** initialize the computation object */
  void Init() {
    
    const char *fname = fx_param_str(NULL, "data", "data.ds");

    // read in the dataset
    Dataset dataset_;
    dataset_.InitFromFile(fname);
    data_.Own(&(dataset_.matrix()));

    // read the orthogonal query range text file
    range_.Init(data_.n_rows());
    RangeReader::ReadRangeData(range_);
    
    // re-initialize boolean flag
    in_range_.Init(data_.n_cols());
    for(index_t i = 0; i < data_.n_cols(); i++) {
      in_range_[i] = false;
    }
  }

  void Init(const Matrix &data, const ArrayList<DRange> &range) {
    data_.Alias(data);
    range_.Copy(range);

    // re-initialize boolean flag
    in_range_.Init(data_.n_cols());
    for(index_t i = 0; i < data_.n_cols(); i++) {
      in_range_[i] = false;
    }
  }

  /** the main computation of naive orthogonal range search */
  void Compute() {

    fx_timer_start(NULL, "naive_search");
    for(index_t i = 0; i < data_.n_cols(); i++) {

      Vector pt;
      bool flag = true;
      data_.MakeColumnVector(i, &pt);
      
      // determine which one of the two cases we have: EXCLUDE, SUBSUME
      // first the EXCLUDE case: when dist is above the upper bound distance
      // of this dimension, or dist is below the lower bound distance of
      // this dimension
      for(index_t d = 0; d < data_.n_rows(); d++) {
	if(pt[d] < range_[d].lo || pt[d] > range_[d].hi) {
	  flag = false;
	  break;
	}
      }
      in_range_[i] = flag;
    }
    fx_timer_stop(NULL, "naive_search");
  }

};

/** Faster orthogonal range search class */
class OrthoRangeSearch {

 public:
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> Tree;

  // getters and setters

  /** get the result of the search by expanding the node list */
  const ArrayList<bool> &get_results() {
    return candidate_points_;
  }

  const Matrix &get_data() const { return data_; }

  const ArrayList<DRange> &get_range() const { return search_range_; }
 
  // interesting functions...

  /** perform the orthogonal range search */
  void Compute() {
    fx_timer_start(NULL, "tree_range_search");
    ortho_range_search(root_, 0);
    fx_timer_stop(NULL, "tree_range_search");
  }

  /** initialization function - to read the data and to construct tree */
  void Init() {

    const char *fname = fx_param_str(NULL, "data", "data.ds");
    int leaflen = fx_param_int(NULL, "leaflen", 20);

    // read in the dataset
    Dataset dataset_;
    dataset_.InitFromFile(fname);
    data_.Own(&(dataset_.matrix()));
    fx_timer_start(NULL, "tree_d");
    root_ = tree::MakeKdTreeMidpoint<Tree>(data_, leaflen, NULL);
    fx_timer_stop(NULL, "tree_d");

    // read in the query range
    search_range_.Init(data_.n_rows());
    RangeReader::ReadRangeData(search_range_);
    
    // initialize candidate nodes and points */
    candidate_points_.Init(data_.n_cols());
    for(index_t i = 0; i < data_.n_cols(); i++) {
      candidate_points_[i] = false;
    }
  }

 private:

  /** flag determining a prune */
  enum PruneStatus {SUBSUME, INCONCLUSIVE, EXCLUDE};

  // member variables
  /** pointer to the dataset */
  Matrix data_;

  /** 
   * orthogonal range to search in: this will be generalized to a list
   * of orthogonal ranges
   */
  ArrayList<DRange> search_range_;
  
  /** the root of the tree */
  Tree *root_;

  /**
   * List of candidate points
   */
  ArrayList<bool> candidate_points_;
  

  // member functions
  
  /** base case */
  void ortho_slow_range_search(Tree *node) {
    PruneStatus prune_flag;

    for(index_t row = node->begin(); row < node->end(); row++) {
      prune_flag = SUBSUME;

      for(index_t d = 0; d < data_.n_rows(); d++) {
	// determine which one of the two cases we have: EXCLUDE, SUBSUME

	DRange search_dir_range = search_range_[d];

	// first the EXCLUDE case: when dist is above the upper bound distance
	// of this dimension, or dist is below the lower bound distance of
	// this dimension
	if(data_.get(d, row) > search_dir_range.hi ||
	   data_.get(d, row) < search_dir_range.lo) {
	  prune_flag = EXCLUDE;
	  break;
	}
      }

      if(prune_flag == SUBSUME) {
	candidate_points_[row] = true;
      }
    }
  }

  /** the workhorse algorithm for fast orthgonal range search */
  void ortho_range_search(Tree *node, int start_dim) {

    PruneStatus prune_flag = SUBSUME;
    
    // loop over each dimension to determine inclusion/exclusion by 
    // determining the lower and the upper bound distance per each dimension 
    // for the given reference node, kn
    for(index_t d = start_dim; d < data_.n_rows(); d++) {
      
      DRange search_dir_range = search_range_[d];
      DRange node_dir_range = node->bound().get(d);

      // determine which one of the three cases we have: EXCLUDE, SUBSUME, or
      // INCONCLUSIVE.
      
      // first the EXCLUDE case: when mindist is above the upper bound 
      // distance of this dimension,  or maxdist is below the lower bound 
      // distance of this dimension
      if(node_dir_range.lo > search_dir_range.hi ||
	 node_dir_range.hi < search_dir_range.lo) {
	return;
      }
      // otherwise, check for SUBSUME case
      else if(search_dir_range.lo <= node_dir_range.lo &&
	      node_dir_range.hi <= search_dir_range.hi) {
      }
      // if any dimension turns out to be inconclusive, then break.
      else {
	start_dim = d;
	prune_flag = INCONCLUSIVE;
	break;
      }
    }
    
    // in case of subsume, then add all points owned by this node to
    // candidates
    if(prune_flag == SUBSUME) {
      for(index_t i = node->begin(); i < node->end(); i++) {
	candidate_points_[i] = true;
      }
      return;
    }
    else if(node->is_leaf()) {
      ortho_slow_range_search(node);
    }
    else {
      ortho_range_search(node->left(), start_dim);
      ortho_range_search(node->right(), start_dim);
    }
  }
};

#endif
