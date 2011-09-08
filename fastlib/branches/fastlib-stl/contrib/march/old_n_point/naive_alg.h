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
    
    arma::mat random_points_;
    // Does this exist?
    arma::colvec random_weights_;
    
    size_t num_points_;
    int num_random_points_;
    size_t tuple_size_;
    
    // These are organized in order of increasing number of random points in 
    // the tuple
    std::vector<int> num_tuples_;
    std::vector<double> weighted_num_tuples_;
    
    SingleMatcher matcher_;
    
    int num_random_;
    
    void ComputeCountsHelper_(std::vector<bool>& permutation_ok, 
                              std::vector<size_t>& points_in_tuple, int k);
      
    
    
  public:
    
    NaiveAlg(arma::mat& data, arma::colvec& weights, arma::mat& random_set, 
             arma::colvec& random_weights,
             arma::mat& matcher_dists, double bandwidth) : 
    matcher_(matcher_dists, bandwidth),
    num_tuples_(matcher_dists.n_cols + 1, 0),
    weighted_num_tuples_(matcher_dists.n_cols + 1, 0.0) { 
      
      data_points_ = data;
      data_weights_ = weights;
      
      random_points_ = random_set;
      random_weights_ = random_weights;
      
      num_points_ = data_points_.n_cols;
      num_random_points_ = random_points_.n_cols;
      tuple_size_ = matcher_dists.n_cols;
      
      num_random_ = 0;
      
    } // constructor
    
    
    void ComputeCounts();
    
    std::vector<int>& num_tuples() {
      return num_tuples_;
    }
    
    std::vector<double>& weighted_num_tuples() {
      return weighted_num_tuples_;
    }
    
    void print_num_tuples() {
      
      
      std::string d_string(tuple_size_, 'D');
      std::string r_string(tuple_size_, 'R');
      
      std::string label_string;
      label_string+=d_string;
      label_string+=r_string;
      
      for (int i = 0; i <= tuple_size_; i++) {
        
        // i is the number of random points in the tuple
        std::string this_string(label_string, i, tuple_size_);
        mlpack::IO::Info << this_string << ": ";
        mlpack::IO::Info << num_tuples_[i] << std::endl;
        
      } // for i
      
    } // print_num_tuples
    
  }; // class
  
  
} // namespace


#endif
