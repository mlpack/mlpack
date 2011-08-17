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
    
    // constructor
    SingleMatcher(arma::mat& matcher_dists, double bandwidth) :
    perms_(matcher_dists.n_cols), 
    lower_bounds_sqr_(matcher_dists.n_rows, matcher_dists.n_cols),
    upper_bounds_sqr_(matcher_dists.n_rows, matcher_dists.n_cols)
    {
      
      
      tuple_size_ = matcher_dists.n_cols;
      
      double half_band = bandwidth / 2.0;
      
      for (index_t i = 0; i < tuple_size_; i++) {
        
        lower_bounds_sqr_(i,i) = 0.0;
        upper_bounds_sqr_(i,i) = 0.0;
        
        for (index_t j = i+1; j < tuple_size_; j++) {
         
          lower_bounds_sqr_(i,j) = (matcher_dists(i,j) - half_band)
                                    * (matcher_dists(i,j) - half_band);
          
          lower_bounds_sqr_(j,i) = (matcher_dists(i,j) - half_band)
          * (matcher_dists(i,j) - half_band);
          
          upper_bounds_sqr_(i,j) = (matcher_dists(i,j) + half_band)
          * (matcher_dists(i,j) + half_band);

          upper_bounds_sqr_(j,i) = (matcher_dists(i,j) + half_band)
          * (matcher_dists(i,j) + half_band);
          
        }
        
      }
      
      num_permutations_ = perms_.num_permutations();
      
    } // constructor
    
    // angle version of the constructor
    SingleMatcher(arma::mat& lower_bounds, arma::mat& upper_bounds) :
    perms_(lower_bounds.n_cols),
    lower_bounds_sqr_(lower_bounds.n_rows, lower_bounds.n_cols),
    upper_bounds_sqr_(upper_bounds.n_rows, upper_bounds.n_cols) {
      
      tuple_size_ = lower_bounds.n_cols;
      
      for (int i = 0; i < tuple_size_; i++) {
        
        lower_bounds_sqr_(i,i) = 0.0;
        upper_bounds_sqr_(i,i) = 0.0;
        
        for (int j = i+1; j < tuple_size_; j++) {
          
          lower_bounds_sqr_(i,j) = lower_bounds(i,j) * lower_bounds(i,j);
          lower_bounds_sqr_(j,i) = lower_bounds_sqr_(i,j);
          
          upper_bounds_sqr_(i,j) = upper_bounds(i,j) * upper_bounds(i,j);
          upper_bounds_sqr_(j,i) = upper_bounds_sqr_(i,j);
          
        }
        
      }
      
      num_permutations_ = perms_.num_permutations();
      
      //lower_bounds_sqr_.print("lower bounds sqr");
      //upper_bounds_sqr_.print("upper bounds sqr");
      
    } // angle constructor
    
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
