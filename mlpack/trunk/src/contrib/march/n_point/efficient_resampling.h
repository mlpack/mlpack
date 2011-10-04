/*
 *  efficient_resampling.h
 *  
 *
 *  Created by William March on 8/24/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef EFFICIENT_RESAMPLING_H
#define EFFICIENT_RESAMPLING_H

#include "node_tuple.h"


namespace npt {
  
  // TODO: figure out how to permute the weights after building trees
  
  class EfficientResampling {
    
  private:
    
    arma::mat data_all_mat_;
    arma::colvec data_all_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    
    NptNode* random_tree_;
    
    std::vector<NptNode*> data_trees_;

    // TODO: what's the right way to handle this?
    // I'm not convinced I actually want pointers here
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
    
    ///////////////// functions ////////////////////////
    int FindRegion_(arma::colvec& col);

    
    void SplitData_();
    
    void BuildTrees_();
    
    
  public:
    
    EfficientResampling(arma::mat& data, arma::colvec& weights,
                        arma::mat& random, arma::colvec& rweights,
                        int num_x_regions, int num_y_regions, int num_z_regions,
                        double box_x_length, double box_y_length, 
                        double box_z_length) :
    // trying to avoid copying data
    data_all_mat_(data.memptr(), data.n_rows, data.n_cols, false), 
    data_all_weights_(weights),
    random_mat_(random.memptr(), random.n_rows, random.n_cols, false), 
    random_weights_(rweights),
    data_trees_(num_resampling_regions_),
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
      
      // I think I still need to do this
      // They will hopefully still exist outside, right?
      for (int i = 0; i < num_resampling_regions_; i++) {
        
        data_mats_[i] = new arma::mat;
        data_weights_[i] = new arma::colvec;
        
      } // for i
      
      // now, find the step sizes
      x_step_ = box_x_length_ / (double)num_x_partitions_;
      y_step_ = box_y_length_ / (double)num_y_partitions_;
      z_step_ = box_z_length_ / (double)num_z_partitions_;
      
      SplitData_();
      
      for (int i = 0; i < num_resampling_regions_; i++) {
        mlpack::IO::Info << "Region " << i <<": " << data_mats_[i]->n_cols;
        mlpack::IO::Info << " points.\n";
      }
      
      
      BuildTrees_();
      
    } // constructor
    
    
    std::vector<arma::mat*>& data_mats() {
      return data_mats_;
    }
    
    arma::mat* data_mat(int i) {
      return data_mats_[i];
    }

    std::vector<arma::colvec*>& data_weights() {
      return data_weights_;
    }
    
    arma::colvec* data_weights(int i) {
      return data_weights_[i];
    }
    
    arma::mat* random_mat() {
      return &random_mat_;
    }
    
    arma::colvec* random_weights() {
      return &random_weights_;
    }
    
    NptNode* random_tree() {
      return random_tree_;
    }
    
    std::vector<NptNode*>& data_trees() {
      return data_trees_;
    }
    
    NptNode* data_tree(int i) {
      return data_trees_[i];
    }
    
    
  }; // class
  
  
} // namespace




#endif

