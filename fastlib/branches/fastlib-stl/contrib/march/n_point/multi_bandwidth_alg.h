/*
 *  multi_bandwidth_alg.h
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MULTI_BANDWIDTH_ALG_H
#define MULTI_BANDWIDTH_ALG_H


#include "node_tuple.h"
#include "multi_matcher.h"
//#include "results_tensor.h"

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
    
    std::vector<int> num_bands_;
    
    
    // need results tensors
    
    int num_prunes_;
    int num_base_cases_;
    
    
    arma::Col<index_t> old_from_new_index_;
    
    NptNode* tree_;
    
    MultiMatcher matcher_;
    
    
    int total_matchers_;
    // indexed by matcher_ind_0 + num_bands[0]*matcher_ind_1 + . . .
    std::vector<int> results_;
    
    // need a matcher
    
    
    ////////////////////// functions //////////////////////////
    
    index_t FindResultsInd_(const std::vector<index_t>& perm_locations);
    
    void FindMatcherInd_(index_t loc, std::vector<index_t>& result);

    
    void BaseCaseHelper_(std::vector<std::vector<index_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<std::vector<index_t> >& perm_locations,
                         std::vector<index_t>& points_in_tuple,
                         int k);
    
    void BaseCase_(NodeTuple& nodes);
    
    bool CanPrune_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
    
  public:
    
    MultiBandwidthAlg(arma::mat& data, arma::colvec& weights, int leaf_size,
                      int tuple_size,
                      const std::vector<double>& min_bands, 
                      const std::vector<double>& max_bands, 
                      const std::vector<int>& num_bands, double bandwidth) :
                      matcher_(min_bands, max_bands, num_bands, bandwidth,
                               tuple_size), num_bands_(num_bands)
    {
      
      // don't forget to initialize the matcher
      
      data_points_ = data;
      
      data_weights_ = weights;
      
      tuple_size_ = tuple_size;
      num_permutations_ = matcher_.num_permutations();
      
      num_points_ = data_points_.n_cols;
      
      leaf_size_ = leaf_size;
      
      num_prunes_ = 0;
      num_base_cases_ = 0;
      
      // initialize the results tensor
      
      tree_ = tree::MakeKdTreeMidpoint<NptNode, double> (data_points_, 
                                                         leaf_size_, 
                                                         old_from_new_index_);
      
      total_matchers_ = 1;
      for (index_t i = 0; i < num_bands.size(); i++) {
        total_matchers_ *= num_bands[i];
      }
      results_.resize(total_matchers_, 0);
      
      
    } // constructor
    
    void OutputResults();
    
    void Compute() {
      
      std::vector<NptNode*> list(tuple_size_, tree_);
      
      NodeTuple nodes(list);
      
      DepthFirstRecursion_(nodes);
      
      std::cout << "Num prunes: " << num_prunes_ << "\n";
      std::cout << "Num base cases: " << num_base_cases_ << "\n";
      // output results here
      
      
    }
    
    
    
    
  }; // class
  

} // namespace

#endif



