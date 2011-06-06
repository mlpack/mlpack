/*
 *  multi_bandwidth_alg.h
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "node_tuple.h"

namespace npt {

  class MultiBandwidthAlg {
    
  private:
    
    // data
    arma::mat data_points_;
    arma::colvec data_weights_;
    
    // general parameters
    index_t num_points_;
    index_t tuple_size_;
    index_t leaf_size_;
    int num_permutations_;
    
    
    // need results tensors
    
    int num_prunes_;
    
    arma::Col<index_t> old_from_new_index_;
    
    NptNode* tree_;
    
    // need a matcher
    
    
    ////////////////////// functions //////////////////////////
    
    void BaseCaseHelper_(std::vector<std::vector<index_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<index_t>& points_in_tuple,
                         int k);
    
    void BaseCase_(NodeTuple& nodes);
    
    bool CanPrune_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
    
  public:
    
    MultiBandwidthAlg(arma::mat& data, arma::colvec& weights, int leaf_size,
                      arma::mat& lower_bds, arma::mat& upper_bds) {
      
      // don't forget to initialize the matcher
      
      data_points_ = data;
      
      data_weights_ = weights;
      
      tuple_size_ = upper_bds.n_cols;
      num_permutations_ = matcher_.num_permutations();
      
      num_points_ = data_points_.n_cols;
      
      leaf_size_ = leaf_size;
      
      num_prunes_ = 0;
      
      // initialize the results tensor
      
      tree_ = tree::MakeKdTreeMidpoint<NptNode, double> (data_points_, 
                                                         leaf_size_, 
                                                         old_from_new_index_);
      
      
      
    } // constructor
    
    void Compute() {
      
      std::vector<NptNode*> list(tuple_size_, tree_);
      
      NodeTuple nodes(list);
      
      DepthFirstRecursion_(nodes);
      
      std::cout << "Num prunes: " << num_prunes_ << "\n";
      
      
      
    }
    
    
    
    
  }; // class
  

} // namespace



