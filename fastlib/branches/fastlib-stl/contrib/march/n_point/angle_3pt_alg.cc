/*
 *  angle_3pt_alg.cc
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "angle_3pt_alg.h"

void npt::Angle3ptAlg::BaseCase_(std::vector<NptNode*>& node_list) {
  
  num_base_cases_++;
  
  // hard coding 3pt here
  for (index_t i = node_list[0]->begin(); i < node_list[0]->end(); i++) {
    
    bool i_is_random = (num_random_ > 0);
    
    arma::colvec vec_i;
    if (i_is_random) {
      vec_i = random_mat_.col(i);
    }
    else {
      vec_i = data_mat_.col(i);
    }
    
    // if i and j are from different sets or different nodes, then the begin 
    // is just the nodes begin
    // otherwise, it's the point after i
    int j_begin = (node_list[0] == node_list[1]) ? i+1 : node_list[1]->begin();
    int j_end = node_list[1]->end();
    
    for (int j = j_begin; j < j_end; j++) {
      
      bool j_is_random = (num_random_ > 1);
      
      arma::colvec vec_j;
      if (j_is_random) {
        vec_j = random_mat_.col(j);
      }
      else {
        vec_j = data_mat_.col(j);
      }
      
      int k_begin = (node_list[1] == node_list[2]) ? j+1 : node_list[2]->begin();
      int k_end = node_list[2]->end();
      
      for (int k = k_begin; k < k_end; k++) {
        
        bool k_is_random = (num_random_ > 2);
        
        arma::colvec vec_k;
        if (k_is_random) {
          vec_k = random_mat_.col(k);
        }
        else {
          vec_k = data_mat_.col(k);
        }
        
        std::vector<int> valid_thetas;
        
        int valid_r1 = matcher_.TestPointTuple(vec_i, vec_j, vec_k,
                                               valid_thetas);
        
        if (valid_r1 >= 0) {
          
          double weight_i = i_is_random ? random_weights_[i] : data_weights_[i];
          double weight_j = j_is_random ? random_weights_[j] : data_weights_[j];
          double weight_k = k_is_random ? random_weights_[k] : data_weights_[k];
          double this_weight = weight_i * weight_j * weight_k;
          
          for (int theta_ind = 0; theta_ind < valid_thetas.size() ; theta_ind++) {
            
            results_[num_random_][valid_r1][theta_ind]++;
            weighted_results_[num_random_][valid_r1][theta_ind]+= this_weight;
             
          } // iterate over valid thetas
          
        } // we found a valid tuple
        
        
      } // for k
      
    } // for j
    
  } // for i
  
  
} // BaseCase_

// returns true if we can prune, false otherwise
bool npt::Angle3ptAlg::CanPrune_(std::vector<NptNode*>& node_list) {
  
  return (!matcher_.TestNodeTuple(node_list[0]->bound(), node_list[1]->bound(),
                                  node_list[2]->bound()));
  
} // CanPrune_

// true if the symmetry is valid, false if we shouldn't consider this list
bool npt::Angle3ptAlg::CheckSymmetry_(std::vector<NptNode*>& node_list) {
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    bool i_is_random = (i < num_random_);
    
    for (index_t j = i+1; j < tuple_size_; j++) {
      
      bool j_is_random = (j < num_random_);
      
      if (node_list[j]->end() <= node_list[i]->begin() && 
          (i_is_random == j_is_random)) {
        return false;
      }
      
    }
    
  }
  
  return true;
  
} // CheckSymmetry_

void npt::Angle3ptAlg::DepthFirstRecursion_(std::vector<NptNode*>& node_list) {
  
  if (CanPrune_(node_list)) {
    
    num_prunes_++;
    
  }
  else {

    bool all_leaves = node_list[0]->is_leaf();
    int split_index = 0;
    int split_count = all_leaves ? 0 : node_list[0]->count();
    
    for (int i = 1; i < tuple_size_; i++) {
      if (!(node_list[i]->is_leaf())) {
        all_leaves = false;
        
        if (node_list[i]->count() > split_count) {
          split_count = node_list[i]->count();
          split_index = i;
        }
        
      }
    } // check for all leaves and which node is largest
    
    if (all_leaves) {
      
      BaseCase_(node_list);
      num_base_cases_++;
      
    } // base case
    else {
      
      NptNode* split_node = node_list[split_index];
      
      node_list[split_index] = split_node->left();
      if (CheckSymmetry_(node_list)) {
        DepthFirstRecursion_(node_list);
      }
      
      node_list[split_index] = split_node->right();
      if (CheckSymmetry_(node_list)) {
        DepthFirstRecursion_(node_list);
      }
      
      node_list[split_index] = split_node;
      
    } // recursion
    
    
  } // can't prune, need to recurse or do a base case
  
  
} // DepthFirstRecursion_

void npt::Angle3ptAlg::OutputResults() {
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  for (int i = 0; i <= tuple_size_; i++) {
    
    // i is the number of random points in the tuple
    std::string this_string(label_string, i, tuple_size_);
    mlpack::IO::Info << this_string << "\n";
    
    for (index_t j = 0; j < results_[i].size(); j++) {
      
      for (index_t k = 0; k < results_[i][j].size(); k++) {
        
        mlpack::IO::Info << "Matcher: ";
        mlpack::IO::Info << "R1: " << r1_[j] << ", ";
        mlpack::IO::Info << "R2: " << (r1_[j] * r2_multiplier_) << ", ";
        mlpack::IO::Info << "theta: " << thetas_[k] << ": ";
        
        mlpack::IO::Info << results_[i][j][k] << "\n";
        
      } // for k
      
    } // for j
    
    mlpack::IO::Info << "\n\n";
    
  } // for i
  
  
  
} // OutputResults()





