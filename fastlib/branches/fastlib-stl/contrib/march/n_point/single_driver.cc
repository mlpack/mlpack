/*
 *  single_driver.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "single_driver.h"


void npt::SingleDriver::Compute() {
  
  for (int i = 0; i < num_resampling_regions_; i++) {
    
    arma::mat& data_mat_i = resampling_class_.data_mat(i);
    NptNode* data_tree_i = resampling_class_.data_tree(i);
    arma::colvec& data_weights_i = resampling_class_.data_weights(i);
    
    for (int num_random = 0; num_random < tuple_size_; num_random++) {
      
      std::vector<arma::mat&> this_comp_mats(tuple_size_);
      std::vector<int> this_comp_multi;
      std::vector<arma::colvec&> this_comp_weights(tuple_size_);
      std::vector<NptNode*> this_comp_trees;
      
      this_comp_mats(0) = data_mat_i;
      this_comp_trees.push_back(data_tree_i);
      this_comp_multi.push_back(1);
      this_comp_weights(0) = data_weights_i;

      if (num_random == 0) {
        
        this_comp_mats(1) = data_mat_i;
        this_comp_mats(2) = data_mat_i;
        this_comp_multi[0] += 2;

        this_comp_weights(1) = data_weights_i;
        this_comp_weights(2) = data_weights_i;

        
      }
      else if (num_random == 1) {
        
        this_comp_mats(1) = data_mat_i;
        this_comp_weights(1) = data_weights_i;

        this_comp_multi[0]++;
        this_comp_mats(2) = resampling_class_.random_mat();
        this_comp_multi.push_back(1);
        this_comp_weights(2) = resampling_class_.random_weights();
        this_comp_trees.push_back(resampling_class_.random_tree());
        
      }
      else {
       // num_random == 2
        this_comp_mats(1) = resampling_class_.random_mat();
        this_comp_mats(2) = resampling_class_.random_mat();
        this_comp_multi.push_back(1);
        this_comp_multi[1]++;
        
        this_comp_weights(1) = resampling_class_.random_weights();
        this_comp_weights(2) = resampling_class_.random_weights();
        
        this_comp_trees.push_back(resampling_class_.random_tree());

        
      }

      // iterate through the matchers
      for (int r1_ind = 0; r1_ind < r1_vec_.size(); r1_ind++) {
        
        for (int theta_ind = 0; theta_ind < theta_vec_.size(); theta_ind++) {
          
          arma::mat& lower_bounds = generator_.lower_matcher(r1_ind, theta_ind);
          arma::mat& upper_bounds = generator_.upper_matcher(r1_ind, theta_ind);
          
          
          // create matcher class
          SingleMatcher matcher(this_comp_mats, this_comp_weights, 
                                lower_bounds, upper_bounds);
          
          // create alg class
          
          GenericNptAlg<SingleMatcher> alg(this_comp_trees, 
                                          this_comp_multi,
                                          matcher);
          
          // run alg class
          alg.Compute();
          
          // process and store results from the matcher
          results_.ProcessResults(i, this_num_random, matcher,
                                  r1_ind, theta_ind);
                      
          
        } // for thetas
        
      } // for r1
      
    }
      
  } // for i

} // Compute()


void npt::SingleDriver::PrintResults() {
  
  results_.PrintResults();
  
} // PrintResults
