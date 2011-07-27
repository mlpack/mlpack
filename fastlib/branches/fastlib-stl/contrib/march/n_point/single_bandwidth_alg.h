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

// TODO: replace SingleNode* with the common type

namespace npt {

  class SingleBandwidthAlg {
    
  private:
    
    // the data and weights
    arma::mat data_points_;
    arma::colvec data_weights_;
    
    arma::mat random_points_;
    arma::colvec random_weights_;

    // input params
    index_t num_points_;
    index_t tuple_size_;
    index_t leaf_size_;
    
    // the matcher
    SingleMatcher matcher_;
    
    // the answer: num_tuples_ is the raw count, weighted is the sum of products
    // of weights of all matching tuples
    std::vector<int> num_tuples_;
    std::vector<double> weighted_num_tuples_;
    
    // the number of times we pruned a tuple
    int num_prunes_;
    int num_base_cases_;
    
    // define the tree
    typedef BinarySpaceTree<DHrectBound<2>, arma::mat> SingleNode; 
    
    arma::Col<index_t> old_from_new_index_data_;
    arma::Col<index_t> old_from_new_index_random_;
    
    SingleNode* data_tree_;
    SingleNode* random_tree_;
    
    int num_random_;
    
    ////////////////////// functions /////////////////////////
    
    //bool CheckSameSet_(int ind1, int ind2);
    
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
    SingleBandwidthAlg(arma::mat& data, arma::colvec weights,
                       arma::mat& random, arma::colvec rweights,
                       index_t leaf_size,
                       arma::mat& matcher_dists, double bandwidth) : 
                       matcher_(matcher_dists, bandwidth),
    num_tuples_(matcher_dists.n_cols + 1, 0),
    weighted_num_tuples_(matcher_dists.n_cols + 1, 0.0)
    {
      
      data_points_ = data;
      data_weights_ = weights;

      random_points_ = random;
      random_weights_ = rweights;

      num_points_ = data_points_.n_cols;
      tuple_size_ = matcher_dists.n_cols;
      
      // TODO: is it worth having a different leaf size for the random points
      leaf_size_ = leaf_size;
      
      num_prunes_ = 0;
      num_base_cases_ = 0;
      
      data_tree_ = tree::MakeKdTreeMidpoint<SingleNode, double>(data_points_, 
                                                                leaf_size_, 
                                                                old_from_new_index_data_);
        
      
      random_tree_ = tree::MakeKdTreeMidpoint<SingleNode, double>(random_points_,
                                                                  leaf_size_,
                                                                  old_from_new_index_random_);
      
      // IMPORTANT: need to permute the weights here
      
      
      
    } // constructor
    
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
    
    
    /**
     * Actually run the algorithm.
     */
    void ComputeCounts() {
            
      for (num_random_ = 0; num_random_ <= tuple_size_; num_random_++) {
      
        std::vector<SingleNode*> node_list(tuple_size_);
        
        for (index_t i = 0; i < num_random_; i++) {
          
          node_list[i] = random_tree_;
          
        }
        for (index_t i = num_random_; i < tuple_size_; i++) {
          
          node_list[i] = data_tree_;
          
        }
        
        DepthFirstRecursion_(node_list);
        
        std::cout << "Num prunes " << num_prunes_ << "\n";
        std::cout << "Num base cases " << num_base_cases_ << "\n";
        
      }
      
    } // ComputeCounts()
    
    
  }; // class
  
  
  
} // namespace

#endif

