/*
 *  naive_resampling.h
 *  
 *
 *  Created by William March on 9/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef NAIVE_RESAMPLING_H
#define NAIVE_RESAMPLING_H

#include "node_tuple.h"

namespace npt {
  
  class NaiveResampling {
    
  private:
    
    arma::mat data_all_mat_;
    arma::colvec data_all_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    NptNode* random_tree_;
    
    arma::mat current_data_mat_;
    arma::colvec current_data_weights_;
    NptNode* current_tree_;
    
    std::vector<arma::mat*> data_mats_;
    std::vector<arma::colvec*> data_weights_;
    
    
    
    int tuple_size_;
    
    int num_x_partitions_;
    int num_y_partitions_;
    int num_z_partitions_;
    
    int num_resampling_regions_;
    
    double x_step_;
    double y_step_;
    double z_step_;
    
    double box_x_length_;
    double box_y_length_;
    double box_z_length_;
    
    int num_points_;
    
    
    int FindRegion_(arma::colvec& col);
    
    void SplitData_();
    
    
  public:
    
    NaiveResampling(arma::mat& data, arma::colvec& weights,
                    arma::mat& random, arma::colvec& rweights,
                    int num_x_regions, int num_y_regions, int num_z_regions,
                    double box_x_length, double box_y_length, 
                    double box_z_length) :
    // trying to avoid copying data
    data_all_mat_(data.memptr(), data.n_rows, data.n_cols, false), 
    data_all_weights_(weights),
    random_mat_(random.memptr(), random.n_rows, random.n_cols, false), 
    random_weights_(rweights),
    data_mats_(num_resampling_regions_),
    data_weights_(num_resampling_regions_),
    num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions),
    num_points_(data.n_cols)
    {
    
      num_x_partitions_ = num_x_regions;
      num_y_partitions_ = num_y_regions;
      num_z_partitions_ = num_z_regions;
      
      box_x_length_ = box_x_length;
      box_y_length_ = box_y_length;;
      box_z_length_ = box_z_length;;
      
      
      tuple_size_ = 3;
      
      // now, find the step sizes
      x_step_ = box_x_length_ / (double)num_x_partitions_;
      y_step_ = box_y_length_ / (double)num_y_partitions_;
      z_step_ = box_z_length_ / (double)num_z_partitions_;
      
      current_tree_ = NULL;
      
      for (int i = 0; i < num_resampling_regions_; i++) {
        
        data_mats_[i] = new arma::mat();
        data_weights_[i] = new arma::colvec();
        
      }
      
      SplitData_();
      
      random_tree_ = new NptNode(random_mat_);
      
    } // constructor
    
    // returns the data for resampling region i
    arma::mat* data_mat(int i);
    
    // IMPORTANT: make sure data_mat() is called first
    arma::colvec* data_weights(int i) {
      return &current_data_weights_;
    }
    
    NptNode* data_tree(int i) {
      return current_tree_;
    }
    
    
    arma::mat* random_mat() {
      return &random_mat_;
    }
    
    arma::colvec* random_weights() {
      return &random_weights_;
    }
    
    // TODO: do I rebuild this every time? 
    NptNode* random_tree() {
      return random_tree_;
    }
      
    
  }; // class
  
  
} // npt


#endif

