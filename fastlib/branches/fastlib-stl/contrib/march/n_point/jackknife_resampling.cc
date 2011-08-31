/*
 *  jackknife_resampling.cc
 *  
 *
 *  Created by William March on 8/24/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "jackknife_resampling.h"

int npt::JackknifeResampling::FindRegion_(arma::colvec& col) {
  
  int h_ind = floor(col(0) / height_step_);
  int w_ind = floor(col(1) / width_step_);
  int d_ind = floor(col(2) / depth_step_);
  
  return (h_ind + num_height_partitions_ * w_ind 
          + num_height_partitions_ * num_width_partitions_ * d_ind);
  
} // FindRegion

int npt::JackknifeResampling::GetRegionID_(std::vector<int>& ids) {
  
  return (ids[0] + num_height_partitions_ * ids[1] 
          + num_height_partitions_ * num_width_partitions_ * ids[2]);
  
}

void npt::JackknifeResampling::SplitData_() {
  
  // iterate through the points and add them to the appropriate region
  
  for (int i = 0; i < num_points_; i++) {
    
    arma::colvec& col_i = data_all_mat_.col(i);
    int region_id = FindRegion_(col_i);
    
    data_mats_[region_id].insert_cols(data_mats_[region_id].n_cols, col_i);
    data_weights_[region_id].insert_rows(data_weights_[region_id].n_rows, 
                                         data_all_weights_(i));
    
  } // for i over points
  
  
} // SplitData


void npt::JackknifeResampling::BuildTrees_() {
  
  random_tree_ = new NptNode(random_mat_, leaf_size_);
  
  // TODO: add old_from_new vectors for weight permutations
  for (int i = 0; i < num_resampling_regions_; i++) {
    
    data_trees_[i] = new NptNode(data_mats_[i], leaf_size_);

  } // for i
  
} // BuildTrees_

void npt::JackknifeResampling::Compute() {
  
  
  // iterate over possible tuples of data matrices
  
  for (int i = 0; i <= num_resampling_regions_; i++) {
  
    std::vector<arma::mat&> this_comp_mats(3);
    std::vector<NptNode*> this_comp_trees;
    std::vector<int> this_comp_multi;
    std::vector<arma::colvec&> this_comp_weights(3);
    arma::col this_region(3);
    
    int this_num_random = 0;
    
    if (i < num_resampling_regions_) {
      // we're adding data
      this_comp_mats[0] = data_mats_[i];
      this_comp_trees.push_back(data_trees_[i]);
      this_comp_multi.push_back(1);
      this_comp_weights[0] = data_weights_[i];
      this_region(0) = i;
    }
    else {
      // we're adding randoms
      this_comp_mats[0] = random_mat_;
      this_comp_trees.push_back(random_tree_);
      this_comp_multi.push_back(1);
      this_comp_weights[0] = random_weights_;
      this_num_random++;
    }
    
    
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
          this_comp_mats[1] = data_mats_[j];
          this_comp_trees.push_back(data_trees_[j]);
          this_comp_multi.push_back(1);
          this_comp_weights[1] = data_weights_[j];
          this_region(1) = j;
        }
        else {
          // we're adding randoms
          this_comp_mats[1] = random_mat_;
          this_comp_trees.push_back(random_tree_);
          this_comp_multi.push_back(1);
          this_comp_weights[1] = random_weights_;
          this_num_random++;
        }
        
      } // not equal to i

      this_region(1) = j;
      
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
            this_comp_mats[2] = data_mats_[k];
            this_comp_trees.push_back(data_trees_[k]);
            this_comp_multi.push_back(1);
            this_comp_weights[2] = data_weights_[k];
            this_region(2) = k;
          }
          else {
            // we're adding randoms
            this_comp_mats[2] = random_mat_;
            this_comp_trees.push_back(random_tree_);
            this_comp_multi.push_back(1);
            this_comp_weights[2] = random_weights_;
            this_num_random++;
          }
          
        } // not equal to i
        
        
        this_region(2) = k;
        
        // create matcher class
        AngleMatcher matcher(this_comp_mats, this_comp_weights, r1_vec_, 
                             r2_mult_, theta_vec_, bin_width_);
        
        // create alg class
        
        GenericNptAlg<AngleMatcher> alg(this_comp_trees, 
                                        this_comp_multi,
                                        matcher);
        
        // run alg class
        
        alg.Compute();
        
        // process and store results from the matcher
      
        // indexed by [r1][theta]
        boost::multi_array<double, 2> this_result = matcher.results();
        
        for (int region_ind = 0; region_ind < num_resampling_regions_; 
             region_ind++) {
        
          if (region_ind != this_region(0) 
              && region_ind != this_region(1)
              && region_ind != this_region(2)) {
            
            // this doesn't actually work, how do I do this?
            results_[region_ind][this_num_random] += this_result;
            
          }
          
        }
        
        // loop over resampling regions that need to be written to
      
      } // for k
    
    } // for j
    
  } // for i
  
  
} // Compute
