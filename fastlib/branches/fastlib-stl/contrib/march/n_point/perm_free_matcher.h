/*
 *  perm_free_matcher.h
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 *
 *  A matcher that does not examine each partition for every prune check.
 */

#ifndef PERM_FREE_MATCHER_H
#define PERM_FREE_MATCHER_H

#include "permutations.h"
#include "node_tuple.h"
#include "fastlib/fastlib.h"

namespace npt {
  
  class PermFreeMatcher {
    
  private:
    
    arma::mat upper_bounds_sq_mat_;
    arma::mat lower_bounds_sq_mat_;
    
    std::vector<double> upper_bounds_sq_;
    std::vector<double> lower_bounds_sq_;
    
    index_t tuple_size_;
    
    // Still need this for exhaustive base cases
    Permutations perms_;
    
    
    ///////////// functions ////////////////////
    
    index_t GetPermutationIndex_(index_t perm_index, index_t pt_index) {
      
      // these needed to be swapped to match matcher code
      return perms_.GetPermutation(perm_index, pt_index);
      
    } // GetPermutation
    
    
    
  public:
    
    PermFreeMatcher(arma::mat& upper, arma::mat& lower) : perms_(upper.n_cols) {
      
      upper_bounds_sq_mat_ = arma::square(upper);
      lower_bounds_sq_mat_ = arma::square(lower);
      
      tuple_size_ = upper.n_cols;
      
      // TODO: check that the matrices are the same size
      
      for (index_t i = 0; i < tuple_size_; i++) {
        for (index_t j = i+1; j < tuple_size_; j++) {
      
          upper_bounds_sq_.push_back(upper_bounds_sq_mat_(i,j));
          lower_bounds_sq_.push_back(lower_bounds_sq_mat_(i,j));
          
        } // for j
      } // for i
      
      // TODO: check that the lists came out the right size
      
      std::sort(upper_bounds_sq_.begin(), upper_bounds_sq_.end());
      std::sort(lower_bounds_sq_.begin(), lower_bounds_sq_.end());
      
    } // constructor
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    bool TestPointPair(double dist_sq, index_t tuple_index_1, 
                       index_t tuple_index_2, 
                       ArrayList<bool>& permutation_ok);
    
    
  }; // class
  
} // namespace

#endif 


