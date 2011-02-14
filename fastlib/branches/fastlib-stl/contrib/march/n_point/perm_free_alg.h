/*
 *  perm_free_alg.h
 *  
 *
 *  Created by William March on 2/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef PERM_FREE_ALG_H
#define PERM_FREE_ALG_H

#include "node_tuple.h"
#include "perm_free_matcher.h"

namespace npt {
  
  class PermFreeAlg {
    
    // data
    arma::mat data_points_;
    arma::colvec data_weights_;
    
    // general parameters
    index_t num_points_;
    index_t tuple_size_;
    index_t leaf_size_;
    
    // matcher
    
    PermFreeMatcher matcher_;
    
    int num_tuples_;
    double weighted_num_tuples_;
    
    int num_prunes_;
    
    arma::Col<index_t> old_from_new_index_;
    
  }; // class
  
  
} // namespace


#endif