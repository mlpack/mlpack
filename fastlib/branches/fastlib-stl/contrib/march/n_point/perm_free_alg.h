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
    
    
  private:
    
    
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
    
    NptNode* tree_;
    
    //////////////// functions //////////////////
    
    void BaseCaseHelper_();
    
    void BaseCase_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
  public:
    
    PermFreeAlg(arma::mat& data, arma::colvec& weights,
                arma::mat& upper_bds, arma::mat& lower_bds,
                int leaf_size) 
    : matcher_(upper_bds, lower_bds) {
      
      data_points_ = data;
      
      data_weights_ = weights;
      
      tuple_size_ = upper_bds.n_cols;
      num_points_ = data_points_.n_cols;
      
      leaf_size_ = leaf_size;
      
      num_tuples_ = 0;
      num_prunes_ = 0;
      weighted_num_tuples_ = 0.0;
      
      tree_ = tree::MakeKdTreeMidpoint<NptNode> (data_points_, leaf_size_, 
                                                 &old_from_new_index_,
                                                 NULL);
      
    } // constructor
    
    void Compute() {
      
      arma::Col<NptNode*> list(tuple_size_);
      list.fill(tree_);
      
      NodeTuple nodes(list);
      
      DepthFirstRecursion_(nodes);
      
    }
    
    
  }; // class
  
  
} // namespace


#endif