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
    index_t leaf_size_;
    
    // the matcher
    SingleMatcher matcher_;
    
    // the answer: num_tuples_ is the raw count, weighted is the sum of products
    // of weights of all matching tuples
    int num_tuples_;
    double weighted_num_tuples_;
    
    // the number of times we pruned a tuple
    int num_prunes_;
    
    // define the tree
    typedef BinarySpaceTree<DHrectBound<2>, arma::mat> SingleNode; 
    
    arma::Col<index_t> old_from_new_index_;
    
    SingleNode* tree_;
    
    
    ////////////////////// functions /////////////////////////
    
    bool CheckNodeList_(std::vector<SingleNode*>& nodes);
    
    void BaseCaseHelper_(std::vector<std::vector<index_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<index_t>& points_in_tuple,
                         int k);
    
    void BaseCase_(std::vector<SingleNode*>& nodes);
    
    void DepthFirstRecursion_(std::vector<SingleNode*>& nodes);
    
    
  public:
    
    /**
     * Requires the matcher bounds, data, parameters
     */
    SingleBandwidthAlg(arma::mat& data, arma::colvec weights, index_t n,
                       index_t leaf_size,
                       arma::mat& lower_bds, arma::mat& upper_bds) : 
                       matcher_(n, lower_bds, upper_bds)
    {
      
      data_points_ = data;
      data_weights_ = weights;
      num_points_ = data_points_.n_cols;
      tuple_size_ = n;
      leaf_size_ = leaf_size;
      
      num_tuples_ = 0;
      weighted_num_tuples_ = 0.0;
      num_prunes_ = 0;
      
      tree_ = tree::MakeKdTreeMidpoint<SingleNode, double>(data_points_, 
                                                           leaf_size_, 
                                                           old_from_new_index_);
        
      // IMPORTANT: need to permute the weights here
      
      
      
    } // constructor
    
    int num_tuples() {
      return num_tuples_;
    } 
    
    double weighted_num_tuples() {
      return weighted_num_tuples_;
    }
    
    
    /**
     * Actually run the algorithm.
     */
    void ComputeCounts() {
            
      std::vector<SingleNode*> node_list(tuple_size_, tree_);
      
      DepthFirstRecursion_(node_list);
      
    } // ComputeCounts()
    
    
    
    
  }; // class
  
  
  
} // namespace

#endif

