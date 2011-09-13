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

#include "fastlib/fastlib.h"

namespace npt {
  
  class NaiveResampling {
    
  private:
    
    arma::mat data_all_mat_;
    arma::colvec data_all_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    
    
    int leaf_size_;
    
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
    
    
    
  public:
    
    EfficientResampling(arma::mat& data, arma::colvec& weights,
                        arma::mat& random, arma::colvec& rweights,
                        int num_x_regions, int num_y_regions, int num_z_regions,
                        double box_x_length, double box_y_length, 
                        double box_z_length
                        int leaf_size) :
    // trying to avoid copying data
    num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions),
    data_all_mat_(data.memptr(), data.n_rows, data.n_cols, false), 
    data_all_weights_(weights),
    random_mat_(random.memptr(), random.n_rows, random.n_cols, false), 
    random_weights_(rweights),
    leaf_size_(leaf_size),
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
      
      
      
    } // constructor
    
    // returns the data for resampling region i
    arma::mat& data_mat(int i);
    
    arma::colvec& data_weights(int i);

    NptNode* data_tree(int i);
    
    arma::mat& random_mat() {
      return random_mat_;
    }
    
    arma::colvec& random_weights() {
      return random_weights_;
    }
    
    // TODO: do I rebuild this every time? 
    NptNode* random_tree();
    
    
  }; // class
  
  
} // npt


#endif

