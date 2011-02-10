/*
 *  single_matcher.h
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 *
 *  A single matcher.  This will use the same matrix formulation as before.
 *
 */

#ifndef SINGLE_MATCHER_H
#define SINGLE_MATCHER_H

#include "permutations.h"

namespace npt {
  
  class SingleMatcher {
    
  private:
    
    // stores the permutations, make sure to always reuse it instead of making
    // more
    Permutations perms_;
    
    // The upper and lower bounds for the matcher, stored in an upper triangular
    // matrix.  They're squared to avoid dealing with square roots of distancess
    arma::mat lower_bounds_sqr_;
    arma::mat upper_bounds_sqr_;
    
    // n
    index_t tuple_size_;
    
    // n!
    index_t num_permutations_;
    
    /////////////////// functions ////////////////////////
    
    /**
     * Just accesses the Permutations class
     */
    index_t GetPermIndex_(index_t perm_index, index_t pt_index) {
      return perms_.GetPermutation(perm_index, pt_index);
    } // GetPermIndex_
    
    
    /**
     * Helper function for checking points or bounds.
     */
    bool CheckDistances_(double dist_sq, index_t ind1, index_t ind2);
    
    
  public:
    
    // Dummy empty constructor
    //SingleMatcher() {}
    
    // constructor
    SingleMatcher(arma::mat& lower_bds, arma::mat& upper_bds) :
    perms_(lower_bds.n_cols) {
      
      
      tuple_size_ = lower_bds.n_cols;
      
      lower_bounds_sqr_ = arma::square(lower_bds);
      upper_bounds_sqr_ = arma::square(upper_bds);
      
      num_permutations_ = perms_.num_permutations();
      
    } // constructor
    
    bool TestPointPair(double dist_sq, index_t tuple_ind_1, index_t tuple_ind_2,
                       std::vector<bool>& permutation_ok);
    
    bool TestHrectPair(const DHrectBound<2>& box1, const DHrectBound<2>& box2,
                       index_t tuple_ind_1, index_t tuple_ind_2,
                       std::vector<bool>& permutation_ok);
    
    index_t num_permutations() {
      return num_permutations_;
    }
    
    
  }; // class
  
} // namespace

#endif
