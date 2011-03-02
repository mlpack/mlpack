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
    int num_permutations_;
    
    // matcher
    
    PermFreeMatcher matcher_;
    
    int num_tuples_;
    double weighted_num_tuples_;
    
    int num_prunes_;
    
    arma::Col<index_t> old_from_new_index_;
    
    NptNode* tree_;
    
    //////////////// functions //////////////////
    
    void BaseCaseHelper_(std::vector<std::vector<index_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<index_t>& points_in_tuple,
                         int k);
    
    void BaseCase_(NodeTuple& nodes);
    
    bool CanPrune_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
  public:
    
    PermFreeAlg(arma::mat& data, arma::colvec& weights, int leaf_size,
                arma::mat& lower_bds, arma::mat& upper_bds)
    : matcher_(upper_bds, lower_bds) {
      
      data_points_ = data;
      
      data_weights_ = weights;
      
      tuple_size_ = upper_bds.n_cols;
      num_permutations_ = matcher_.num_permutations();
      
      num_points_ = data_points_.n_cols;
      
      leaf_size_ = leaf_size;
      
      num_tuples_ = 0;
      num_prunes_ = 0;
      weighted_num_tuples_ = 0.0;
      
      tree_ = tree::MakeKdTreeMidpoint<NptNode, double> (data_points_, 
                                                         leaf_size_, 
                                                         old_from_new_index_);
      
    } // constructor
    
    double weighted_num_tuples() const {
      return weighted_num_tuples_;
    }
    
    int num_tuples() const {
      return num_tuples_;
    } // num_tuples
    
    void Compute() {
      
      std::vector<NptNode*> list(tuple_size_, tree_);
      
      NodeTuple nodes(list);
      
      DepthFirstRecursion_(nodes);
      
    }
    
    
  }; // class
  
  
} // namespace


#endif