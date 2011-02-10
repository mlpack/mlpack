/*
 *  naive_alg.h
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 *  A naive implementation for comparison purposes.  
 */

#ifndef NAIVE_ALG_H
#define NAIVE_ALG_H

#include "single_matcher.h"


namespace npt {
  
  class NaiveAlg {
    
  private:
    
    arma::mat data_points_;
    arma::colvec data_weights_;
    
    index_t num_points_;
    index_t tuple_size_;
    
    int num_tuples_;
    double weighted_num_tuples_;
    
    SingleMatcher matcher_;
    
    void ComputeCountsHelper_(std::vector<bool>& permutation_ok, 
                              std::vector<index_t>& points_in_tuple, int k);
      
    
    
  public:
    
    NaiveAlg(arma::mat& data, arma::colvec& weights, arma::mat& lower_bds, 
             arma::mat& upper_bds) : 
    matcher_(lower_bds, upper_bds) {
      
      data_points_ = data;
      data_weights_ = weights;
      
      num_points_ = data_points_.n_cols;
      tuple_size_ = lower_bds.n_cols;
      
      
      num_tuples_ = 0;
      weighted_num_tuples_ = 0.0;
      
    } // constructor
    
    
    void ComputeCounts();
    
    int num_tuples() {
      return num_tuples_;
    }
    
    double weighted_num_tuples() {
      return weighted_num_tuples_;
    }
    
    
  }; // class
  
  
} // namespace


#endif
