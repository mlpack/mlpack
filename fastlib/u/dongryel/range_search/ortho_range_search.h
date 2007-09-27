#ifndef ORTHO_RANGE_SEARCH_H
#define ORTHO_RANGE_SEARCH_H

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
  DHrectBound<2> range_;
  
 public:

  NaiveOrthoRangeSearch() {}

  ~NaiveOrthoRangeSearch() {}
  
  /** initialize the computation object */
  void Init() {
    
    const char *fname = fx_param_str(NULL, "data", NULL);
    
    // read in the dataset
    Dataset dataset_;
    dataset_.InitFromFile(fname);
    data_.Own(&(dataset_.matrix()));

    // re-initialize boolean flag
    for(index_t i = 0; i < data_.n_cols(); i++) {
      in_range_[i] = false;
    }
  }

  /** the main computation of naive orthogonal range search */
  void Compute() {

    for(index_t i = 0; i < data_.n_cols(); i++) {

      Vector pt;
      data_.MakeColumnVector(i, &pt);
      
      // determine which one of the two cases we have: EXCLUDE, SUBSUME
      // first the EXCLUDE case: when dist is above the upper bound distance
      // of this dimension, or dist is below the lower bound distance of
      // this dimension
      if(range_.Contains(pt)) {
	in_range_[i] = true;
      }
    }
  }

};

/** Faster orthogonal range search class */
class OrthoRangeSearch {

  
};

#endif
