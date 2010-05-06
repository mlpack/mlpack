/*
 *  perm_free_matcher.h
 *  
 *
 *  Created by William March on 4/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef PERM_FREE_MATCHER_H
#define PERM_FREE_MATCHER_H


#include "n_point_nodes.h"
#include "n_point_impl.h"

// TODO: still need permutations for the base case, I think

class PermFreeMatcher {
  
private:
    
  Matrix upper_bounds_sq_mat_;
  
  ArrayList<double> upper_bounds_sq_;
  
  index_t tuple_size_;
  
  Permutations perms_;
  int num_permutations_;
  
  
public:
  
  index_t GetPermutationIndex_(index_t perm_index, index_t pt_index) {
    
    // these needed to be swapped to match matcher code
    return perms_.GetPermutation(perm_index, pt_index);
    
  } // GetPermutation
  
  int num_permutations() {
    return num_permutations_;
  }
  
  // assuming the input isn't squared or sorted
  void Init(const Matrix& mat_in, ArrayList<double>& upper_in,
            index_t size_in) {
    
    upper_bounds_sq_mat_.Init(mat_in.n_rows(), mat_in.n_cols());
    for (index_t i = 0; i < mat_in.n_rows(); i++) {
      for (index_t j = 0; j < mat_in.n_cols(); j++) {
        upper_bounds_sq_mat_.set(i, j, mat_in.get(i, j) * mat_in.get(i, j));
      }
    }
    
    tuple_size_ = size_in;
    
    upper_bounds_sq_.Init(upper_in.size());
    
    for (index_t i = 0; i < upper_in.size(); i++) {
      
      upper_bounds_sq_[i] = upper_in[i] * upper_in[i];
      
    } // for i
    
    std::sort(upper_bounds_sq_.begin(), upper_bounds_sq_.end());
  
    perms_.Init(tuple_size_);
    num_permutations_ = perms_.num_perms();
    
    
  } // Init()
  
  bool TestPointPair(double dist_sq, index_t tuple_index_1, 
                     index_t tuple_index_2, 
                     ArrayList<bool>& permutation_ok);
  
  
  int CheckNodes(NodeTuple& nodes);
  
  
}; // PermFreeMatcher



#endif


