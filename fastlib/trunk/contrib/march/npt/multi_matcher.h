/*
 *  multi_matcher.h
 *  
 *
 *  Created by William March on 3/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MULTI_MATCHER_H
#define MULTI_MATCHER_H

#include "fastlib/fastlib.h"
#include "n_point_impl.h"

class MultiMatcher {
  
private:
  
    
  ////////// variables /////////////
  
  ArrayList<double> distances_;
  int num_bins_;
  
  Permutations perms_;
  int num_permutations_;
  
  int tuple_size_;
  
public:

  index_t GetPermutationIndex_(index_t perm_index, index_t pt_index) {
    
    // these needed to be swapped to match matcher code
    return perms_.GetPermutation(perm_index, pt_index);
    
  } // GetPermutation
  
  int num_permutations() {
    return num_permutations_;
  }
  
  int num_bins() {
    return num_bins_;
  }
  
  double max_dist() {
    return distances_[num_bins_ - 1];
  }
  
  ArrayList<double>& distances() {
    return distances_;
  }
  
  bool TestPointPair(double dist_sq, index_t tuple_index_1, 
                     index_t tuple_index_2, 
                     ArrayList<bool>& permutation_ok,
                     ArrayList<GenMatrix<index_t> >& permutation_ranges);
  

  void Init(ArrayList<double>& dists, int n) {
    
    tuple_size_ = n;
    
    // TODO: are these squared?
    distances_.InitCopy(dists);
    num_bins_ = distances_.size();
    
    perms_.Init(tuple_size_);
    num_permutations_ = perms_.num_perms();
    
  } // Init()
  
}; // class MultiMatcher


#endif 

