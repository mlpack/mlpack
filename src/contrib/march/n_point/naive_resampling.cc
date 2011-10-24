/*
 *  naive_resampling.cc
 *  
 *
 *  Created by William March on 9/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "naive_resampling.h"

int npt::NaiveResampling::FindRegion_(arma::colvec& col) {
  
  int x_ind = floor(col(0) / x_step_);
  int y_ind = floor(col(1) / y_step_);
  int z_ind = floor(col(2) / z_step_);
  
  return (x_ind + num_x_partitions_ * y_ind 
          + num_x_partitions_ * num_y_partitions_ * z_ind);
  
} // FindRegion

void npt::NaiveResampling::SplitData_() {
  
  // iterate through the points and add them to the appropriate region
  
  for (int i = 0; i < num_points_; i++) {
    
    arma::colvec col_i = data_all_mat_.col(i);
    int region_id = FindRegion_(col_i);
    
    // TODO: is this the most efficient way to do this?
    data_mats_[region_id]->insert_cols(data_mats_[region_id]->n_cols, col_i);
    data_weights_[region_id]->insert_rows(data_weights_[region_id]->n_rows, 
                                         data_all_weights_(i));
    
  } // for i over points
  
  
} // SplitData


arma::mat* npt::NaiveResampling::data_mat(int i) {
  
  // make sure i >= 0, <= num_resampling_regions
  
  current_data_mat_.reset();
  if (current_tree_ != NULL) {
    delete current_tree_;
  }
  
  // need to stick all but the ith data mat together
  for (int j = 0; j < num_resampling_regions_; j++) {
    
    if (j != i) {
      current_data_mat_.insert_cols(current_data_mat_.n_cols, *(data_mats_[j]));
      current_data_weights_.insert_rows(current_data_weights_.n_rows,
                                        *(data_weights_[j]));
    }
    
  } // for j
  
  current_tree_ = new NptNode(current_data_mat_);
  
  return &current_data_mat_;
  
} // data_mat


