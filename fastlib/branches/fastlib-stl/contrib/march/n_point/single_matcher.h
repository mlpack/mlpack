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

/*
 *  Takes in a single matcher, specified either by upper and lower bound
 *  matrices or a matrix of distances and a bandwidth
 *  Called by the generic algorithm to do prune checks 
 *  and base cases.
 *
 *  Needs access to the data.  Right now, I'm using a reference to a 
 *  vector of references to matrices.  I don't think this works.
 *  Other options: 
 *  - aliasing (like the neighbor search code) 
 *  - pointers to memory that's held by the resampling class.
 */


#ifndef SINGLE_MATCHER_H
#define SINGLE_MATCHER_H

#include "permutations.h"
#include "node_tuple.h"

namespace npt {
  
  class SingleMatcher {
    
  private:
    
    
    // the data and weights
    std::vector<arma::mat*> data_mat_list_;
    std::vector<arma::colvec*> data_weights_list_;
    
    std::vector<std::vector<size_t>*> old_from_new_list_;
    
    
    // stores the permutations, make sure to always reuse it instead of making
    // more
    Permutations perms_;
    
    // The upper and lower bounds for the matcher, stored in an upper triangular
    // matrix.  They're squared to avoid dealing with square roots of distancess
    arma::mat lower_bounds_sqr_;
    arma::mat upper_bounds_sqr_;
    
    int results_;
    double weighted_results_;
    
    // n
    int tuple_size_;
    
    int num_base_cases_;
    
    // n!
    int num_permutations_;
    
    /////////////////// functions ////////////////////////
    
    /**
     * Just accesses the Permutations class
     */
    int GetPermIndex_(int perm_index, int pt_index) {
      return perms_.GetPermutation(perm_index, pt_index);
    } // GetPermIndex_
    
    
    /**
     * Helper function for checking points or bounds.
     */
    bool CheckDistances_(double dist_sq, int ind1, int ind2);
    
    
    bool TestPointPair_(double dist_sq, int tuple_ind_1, int tuple_ind_2,
                        std::vector<bool>& permutation_ok);
    
    bool TestHrectPair_(const DHrectBound<2>& box1, const DHrectBound<2>& box2,
                        int tuple_ind_1, int tuple_ind_2,
                        std::vector<bool>& permutation_ok);
    
    void BaseCaseHelper_(NodeTuple& nodes,
                         std::vector<bool>& permutation_ok,
                         std::vector<int>& points_in_tuple,
                         int k);
    
    
  public:
    
    
    // constructor
    SingleMatcher(std::vector<arma::mat*>& data_in, 
                  std::vector<arma::colvec*>& weights_in,
                  std::vector<std::vector<size_t>*>& old_from_new_in,
                  arma::mat& matcher_dists, double bandwidth) :
    data_mat_list_(data_in), data_weights_list_(weights_in),
    old_from_new_list_(old_from_new_in),
    perms_(matcher_dists.n_cols), 
    lower_bounds_sqr_(matcher_dists.n_rows, matcher_dists.n_cols),
    upper_bounds_sqr_(matcher_dists.n_rows, matcher_dists.n_cols)
    {
      
      results_ = 0;
      weighted_results_ = 0.0;
      
      tuple_size_ = matcher_dists.n_cols;
      
      double half_band = bandwidth / 2.0;
      
      for (int i = 0; i < tuple_size_; i++) {
        
        lower_bounds_sqr_(i,i) = 0.0;
        upper_bounds_sqr_(i,i) = 0.0;
        
        for (int j = i+1; j < tuple_size_; j++) {
         
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
    SingleMatcher(std::vector<arma::mat*>& data_in, 
                  std::vector<arma::colvec*>& weights_in,
                  std::vector<std::vector<size_t>*>& old_from_new_in,
                  arma::mat& lower_bounds, arma::mat& upper_bounds) :
    data_mat_list_(data_in), data_weights_list_(weights_in),
    old_from_new_list_(old_from_new_in),
    perms_(lower_bounds.n_cols),
    lower_bounds_sqr_(lower_bounds.n_rows, lower_bounds.n_cols),
    upper_bounds_sqr_(upper_bounds.n_rows, upper_bounds.n_cols)
    {
      
      results_ = 0;
      weighted_results_ = 0.0;
      
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
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    void ComputeBaseCase(NodeTuple& nodes);
    
    int results() {
      return results_;
    }

    double weighted_results() {
      return weighted_results_;
    }
    
    int tuple_size() {
      return tuple_size_;
    }
    
    int num_permutations() {
      return num_permutations_;
    }
    
    void OutputResults();
    
    
  }; // class
  
} // namespace

#endif
