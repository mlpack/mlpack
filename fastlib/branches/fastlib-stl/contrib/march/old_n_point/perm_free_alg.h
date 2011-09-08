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
//#include "old_node_tuple.h"
#include "perm_free_matcher.h"

namespace npt {
  
  class PermFreeAlg {
    
    
  private:
    
    
    // data
    arma::mat data_points_;
    arma::colvec data_weights_;
    
    arma::mat random_points_;
    arma::colvec random_weights_;
    
    // general parameters
    size_t num_points_;
    size_t tuple_size_;
    size_t leaf_size_;
    int num_permutations_;
    
    int num_random_;
    
    // matcher
    
    PermFreeMatcher matcher_;
    
    std::vector<int> num_tuples_;
    std::vector<double> weighted_num_tuples_;
    
    int num_prunes_;
    int num_base_cases_;
    
    //arma::Col<size_t> old_from_new_index_;
    //arma::Col<size_t> old_from_new_index_random_;
    
    std::vector<size_t> old_from_new_index_;
    std::vector<size_t> old_from_new_index_random_;
    
    NptNode* tree_;
    NptNode* random_tree_;
    
    //////////////// functions //////////////////
    
    void BaseCaseHelper_(std::vector<std::vector<size_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<size_t>& points_in_tuple,
                         int k);
    
    void BaseCase_(NodeTuple& nodes);
    
    bool CanPrune_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
  public:
    
    PermFreeAlg(arma::mat& data, arma::colvec& weights, 
                arma::mat& random, arma::colvec& rweights,
                int leaf_size,
                arma::mat& matcher_dists, double bandwidth)
    : matcher_(matcher_dists, bandwidth),
      num_tuples_(matcher_dists.n_cols+1, 0),
      weighted_num_tuples_(matcher_dists.n_cols+1, 0)
    {
      
      data_points_ = data;
      
      data_weights_ = weights;
      
      random_points_ = random;
      random_weights_ = rweights;
      
      tuple_size_ = matcher_dists.n_cols;
      num_permutations_ = matcher_.num_permutations();
      
      num_points_ = data_points_.n_cols;
      
      leaf_size_ = leaf_size;
      
      num_prunes_ = 0;
      num_base_cases_ = 0;
      
      //tree_ = mlpack::tree::MakeKdTreeMidpoint<NptNode, double> (data_points_, 
        //                                                 leaf_size_, 
          //                                               old_from_new_index_);
      
      //random_tree_ = mlpack::tree::MakeKdTreeMidpoint<NptNode, double>(random_points_,
        //                                                       leaf_size_,
          //                                                     old_from_new_index_random_);
      
      tree_ = new NptNode(data_points_, leaf_size_, old_from_new_index_);
      random_tree_ = new NptNode(random_points_, leaf_size_, 
                                 old_from_new_index_random_);
      
    } // constructor
    
    std::vector<double>& weighted_num_tuples() {
      return weighted_num_tuples_;
    }
    
    std::vector<int>& num_tuples() {
      return num_tuples_;
    } // num_tuples
    
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
    
    
    
    void Compute() {
      
      
      for (num_random_ = 0; num_random_ <= tuple_size_; num_random_++) {
        
        std::vector<NptNode*> node_list(tuple_size_);
        
        for (int i = 0; i < num_random_; i++) {
          
          node_list[i] = random_tree_;
          
        }
        for (int i = num_random_; i < tuple_size_; i++) {
          
          node_list[i] = tree_;
          
        }
        
        NodeTuple nodes(node_list, num_random_);
        
        DepthFirstRecursion_(nodes);
        
      } // for num_random
      
      std::cout << "Num prunes: " << num_prunes_ << "\n";
      std::cout << "Num base cases: " << num_base_cases_ << "\n";
      
    }
    
  }; // class
  
  
} // namespace


#endif