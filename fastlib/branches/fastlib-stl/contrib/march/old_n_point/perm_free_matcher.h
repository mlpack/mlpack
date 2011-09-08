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
//#include "old_node_tuple.h"
#include "fastlib/fastlib.h"

namespace npt {
  
  class PermFreeMatcher {
    
  private:
    
    // data
    arma::mat data_mat_;
    arma::colvec data_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    
    std::vector<int> results_;
    std::vector<double> weighted_results_;
    
    
    arma::mat upper_bounds_sqr_mat_;
    arma::mat lower_bounds_sqr_mat_;
    
    std::vector<double> upper_bounds_sqr_;
    std::vector<double> lower_bounds_sqr_;
    
    size_t tuple_size_;
    
    int num_random_;
    
    // Still need this for exhaustive base cases
    Permutations perms_;
    
    
    ///////////// functions ////////////////////
    
    size_t GetPermutationIndex_(size_t perm_index, size_t pt_index) {
      
      // these needed to be swapped to match matcher code
      return perms_.GetPermutation(perm_index, pt_index);
      
    } // GetPermutation
    
    bool TestPointPair_(double dist_sq, size_t tuple_index_1, 
                        size_t tuple_index_2, 
                        std::vector<bool>& permutation_ok);
    
    void BaseCaseHelper_(std::vector<std::vector<size_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<size_t>& points_in_tuple,
                         int k);
    
    
    
  public:
    
    PermFreeMatcher(arma::mat& data, arma::colvec& weights,
                    arma::mat& random, arma::colvec& rweights,
                    arma::mat& matcher_dists, double bandwidth) 
    : perms_(matcher_dists.n_cols),
    lower_bounds_sqr_mat_(matcher_dists.n_rows, matcher_dists.n_cols),
    upper_bounds_sqr_mat_(matcher_dists.n_rows, matcher_dists.n_cols),
    data_mat_(data), data_weights_(weights),
    random_mat_(random), random_weights_(rweights),
    results_(matcher_dists.n_rows + 1),
    weighted_results_(matcher_dists.n_rows + 1)
    {
      
      double half_band = bandwidth / 2.0;
      
      tuple_size_ = matcher_dists.n_rows;
      
      //std::cout << "half band: " << half_band << "\n";
      
      //matcher_dists.print("matcher dists");
      
      for (size_t i = 0; i < tuple_size_; i++) {
        
        lower_bounds_sqr_mat_(i,i) = 0.0;
        upper_bounds_sqr_mat_(i,i) = 0.0;
        
        for (size_t j = i+1; j < tuple_size_; j++) {
          
          lower_bounds_sqr_mat_(i,j) = (matcher_dists(i,j) - half_band)
          * (matcher_dists(i,j) - half_band);
          
          lower_bounds_sqr_mat_(j,i) = (matcher_dists(i,j) - half_band)
          * (matcher_dists(i,j) - half_band);
          
          upper_bounds_sqr_mat_(i,j) = (matcher_dists(i,j) + half_band)
          * (matcher_dists(i,j) + half_band);
          
          upper_bounds_sqr_mat_(j,i) = (matcher_dists(i,j) + half_band)
          * (matcher_dists(i,j) + half_band);
          
        }
        
      }
      
      
      
      //lower_bounds_sqr_mat_.print("Lower");
      //upper_bounds_sqr_mat_.print("Upper");
      
      
      for (size_t i = 0; i < tuple_size_; i++) {
        for (size_t j = i+1; j < tuple_size_; j++) {
      
          upper_bounds_sqr_.push_back(upper_bounds_sqr_mat_(i,j));
          lower_bounds_sqr_.push_back(lower_bounds_sqr_mat_(i,j));
          
        } // for j
      } // for i
      
      // TODO: check that the lists came out the right size
      
      std::sort(upper_bounds_sqr_.begin(), upper_bounds_sqr_.end());
      std::sort(lower_bounds_sqr_.begin(), lower_bounds_sqr_.end());
      
      
    } // constructor
    
    int num_permutations() const {
      return perms_.num_permutations();
    }
    
    std::vector<int>& results() {
     return results_ 
    }

    std::vector<double>& weighted_results() {
      return weighted_results_ 
    }
    
    void set_num_random(int n) {
      num_random_ = n;
    }
    
    
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    void BaseCase(NodeTuple& nodes);
    
  }; // class
  
} // namespace

#endif 


