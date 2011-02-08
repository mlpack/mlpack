/*
 *  single_bandwidth_alg.h
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 *
 *  Contains the single-bandwidth implementation algorithm class.
 *
 *  This is basically the same as the old auton code.
 */

#ifndef SINGLE_BANDWIDTH_ALG_H
#define SINGLE_BANDWIDTH_ALG_H

#include "single_matcher.h"

namespace npt {

  class SingleBandwidthAlg {
    
  private:
    
    // the data and weights
    arma::mat data_points_;
    arma::colvec data_weights_;

    // input params
    index_t num_points_;
    index_t tuple_size_;
    
    // the matcher
    SingleMatcher matcher_;
    
    // the answer: num_tuples_ is the raw count, weighted is the sum of products
    // of weights of all matching tuples
    int num_tuples_;
    double weighted_num_tuples_;
    
    // the number of times we pruned a tuple
    int num_prunes_;
    
    NPointNode* tree_;
    
  public:
    
    
    
    
    
  }; // class
  
  
  
} // namespace

#endif

