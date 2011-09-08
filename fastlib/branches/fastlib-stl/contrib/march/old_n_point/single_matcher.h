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
#include "node_tuple.h"

namespace npt {
  
  class SingleMatcher {
    
  private:
    
    
    // the data and weights
    arma::mat data_mat_;
    arma::colvec data_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    
    
    // stores the permutations, make sure to always reuse it instead of making
    // more
    Permutations perms_;
    
    // The upper and lower bounds for the matcher, stored in an upper triangular
    // matrix.  They're squared to avoid dealing with square roots of distancess
    arma::mat lower_bounds_sqr_;
    arma::mat upper_bounds_sqr_;
    
    std::vector<int> results_;
    std::vector<double> weighted_results_;
    
    int num_random_;
    
    // n
    size_t tuple_size_;
    
    int num_base_cases_;
    
    // n!
    size_t num_permutations_;
    
    /////////////////// functions ////////////////////////
    
    /**
     * Just accesses the Permutations class
     */
    size_t GetPermIndex_(size_t perm_index, size_t pt_index) {
      return perms_.GetPermutation(perm_index, pt_index);
    } // GetPermIndex_
    
    
    /**
     * Helper function for checking points or bounds.
     */
    bool CheckDistances_(double dist_sq, size_t ind1, size_t ind2);
    
    
    bool TestPointPair_(double dist_sq, size_t tuple_ind_1, size_t tuple_ind_2,
                        std::vector<bool>& permutation_ok);
    
    bool TestHrectPair_(const DHrectBound<2>& box1, const DHrectBound<2>& box2,
                        size_t tuple_ind_1, size_t tuple_ind_2,
                        std::vector<bool>& permutation_ok);
    
    void BaseCaseHelper_(std::vector<std::vector<size_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<size_t>& points_in_tuple,
                         int k);
    
    
  public:
    
    
    // constructor
    SingleMatcher(arma::mat& data, arma::colvec& weights,
                  arma::mat& random, arma::colvec& rweights,
                  arma::mat& matcher_dists, double bandwidth) :
    data_mat_(data), data_weights_(weights),
    random_mat_(random), random_weights_(rweights),
    perms_(matcher_dists.n_cols), 
    lower_bounds_sqr_(matcher_dists.n_rows, matcher_dists.n_cols),
    upper_bounds_sqr_(matcher_dists.n_rows, matcher_dists.n_cols),
    results_(matcher_dists.n_rows + 1, 0),
    weighted_results_(matcher_dists.n_rows + 1, 0.0)
    {
      
      
      tuple_size_ = matcher_dists.n_cols;
      
      double half_band = bandwidth / 2.0;
      
      for (size_t i = 0; i < tuple_size_; i++) {
        
        lower_bounds_sqr_(i,i) = 0.0;
        upper_bounds_sqr_(i,i) = 0.0;
        
        for (size_t j = i+1; j < tuple_size_; j++) {
         
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
    SingleMatcher(arma::mat& data, arma::colvec& weights,
                  arma::mat& random, arma::colvec& rweights,
                  arma::mat& lower_bounds, arma::mat& upper_bounds) :
    perms_(lower_bounds.n_cols),
    data_mat_(data), data_weights_(weights),
    random_mat_(random), random_weights_(rweights),
    lower_bounds_sqr_(lower_bounds.n_rows, lower_bounds.n_cols),
    upper_bounds_sqr_(upper_bounds.n_rows, upper_bounds.n_cols),
    results_(lower_bounds.n_rows + 1, 0),
    weighted_results_(lower_bounds.n_rows + 1, 0.0)
    {
      
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
    
    std::vector<int>& results() {
      return results_;
    }

    std::vector<double>& weighted_results() {
      return weighted_results_;
    }
    
    int tuple_size() {
      return tuple_size_;
    }
    
    void set_num_random(int n) {
      num_random_ = n;
    }
    
    
    size_t num_permutations() {
      return num_permutations_;
    }
    
    void OutputResults();
    
    
  }; // class
  
} // namespace

#endif
