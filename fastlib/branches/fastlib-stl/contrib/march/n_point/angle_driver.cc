/*
 *  angle_driver.cc
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "angle_driver.h"


void npt::AngleDriver::Compute() {
  
  for (int i = 0; i <= num_resampling_regions_; i++) {
    
    // TODO: make sure this doesn't copy memory
    std::vector<arma::mat*> this_comp_mats(3);
    std::vector<NptNode*> this_comp_trees;
    std::vector<int> this_comp_multi;
    std::vector<arma::colvec*> this_comp_weights(3);
    std::vector<int> this_region(3);
    
    int this_num_random = 0;
    
    if (i < num_resampling_regions_) {
      // we're adding data
      this_comp_mats[0] = resampling_class_.data_mat(i);
      this_comp_trees.push_back(resampling_class_.data_tree(i));
      this_comp_multi.push_back(1);
      this_comp_weights[0] = resampling_class_.data_weights(i);
    }
    else {
      // we're adding randoms
      this_comp_mats[0] = resampling_class_.random_mat();
      this_comp_trees.push_back(resampling_class_.random_tree());
      this_comp_multi.push_back(1);
      this_comp_weights[0] = resampling_class_.random_weights();
      this_num_random++;
    }
    
    this_region[0] = i;
    
    
    for (int j = i; j <= num_resampling_regions_; j++) {
      
      if (j == i) {
        this_comp_multi.back()++;
        this_comp_mats[1] = this_comp_mats[0];
        this_comp_weights[1] = this_comp_weights[0];
        if (j == num_resampling_regions_) {
          this_num_random++;
        }
        
      }
      else {
        
        if (j < num_resampling_regions_) {
          // we're adding data
          this_comp_mats[1] = resampling_class_.data_mat(j);
          this_comp_trees.push_back(resampling_class_.data_tree(j));
          this_comp_multi.push_back(1);
          this_comp_weights[1] = resampling_class_.data_weights(j);
        }
        else {
          // we're adding randoms
          this_comp_mats[1] = resampling_class_.random_mat();
          this_comp_trees.push_back(resampling_class_.random_tree());
          this_comp_multi.push_back(1);
          this_comp_weights[1] = resampling_class_.random_weights();
          this_num_random++;
        } 
        
      } // not equal to i
      
      this_region[1] = j;
      
      for (int k = j; k <= num_resampling_regions_; k++) {
        
        if (k == j) {
          this_comp_multi.back()++;
          this_comp_mats[2] = this_comp_mats[1];
          this_comp_weights[2] = this_comp_weights[1];
          
          if (k == num_resampling_regions_) {
            this_num_random++;
          }
          
        }
        else {
          
          if (k < num_resampling_regions_) {
            // we're adding data
            this_comp_mats[2] = resampling_class_.data_mat(k);
            this_comp_trees.push_back(resampling_class_.data_tree(k));
            this_comp_multi.push_back(1);
            this_comp_weights[2] = resampling_class_.data_weights(k);
          }
          else {
            // we're adding randoms
            this_comp_mats[2] = resampling_class_.random_mat();
            this_comp_trees.push_back(resampling_class_.random_tree());
            this_comp_multi.push_back(1);
            this_comp_weights[2] = resampling_class_.random_weights();
            this_num_random++;
          }
          
        } // not equal to j
        
        
        this_region[2] = k;
        
        // create matcher class
        AngleMatcher matcher(this_comp_mats, this_comp_weights, r1_vec_, 
                             r2_multiplier_, theta_vec_, bin_size_);
        
        // create alg class
        
        GenericNptAlg<AngleMatcher> alg(this_comp_trees, 
                                        this_comp_multi,
                                        matcher);
        
        // run alg class
        alg.Compute();
        
        // process and store results from the matcher
        results_.ProcessResults(this_region, this_num_random, matcher);
        
      } // for k
      
    } // for j
    
  } // for i
  
  
} // Compute()


void npt::AngleDriver::PrintResults() {
  
  results_.PrintResults();
  
} // OutputResults




