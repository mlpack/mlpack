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

#include "fastlib/fastlib.h"



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
    // convention: last one is the random
    std::vector<arma::mat*> data_mats_;
    std::vector<arma::colvec*> data_weights_;
    
    int leaf_size_;
    
    int tuple_size_;
    
    int num_resampling_regions_;
    
    int num_height_partitions_;
    int num_width_partitions_;
    int num_depth_partitions_;
    
    double height_step_;
    double width_step_;
    double depth_step_;

    // IMPORTANT: assuming the data live in a cube
    double data_box_length_;
    
    ///////////////// functions ////////////////////////
    int FindRegion_(arma::colvec& col);

    
    void SplitData_();
    
    void BuildTrees_();
    
    
  public:
    
    EfficientResampling(arma::mat& data, arma::colvec& weights,
                        arma::mat& random, arma::colvec& rweights,
                        int num_regions, double box_length,
                        int leaf_size) :
    // trying to avoid copying data
    data_all_mat_(data.memptr(), data.n_rows, data.n_cols, false), 
    data_all_weights_(weights),
    random_mat_(random.memptr(), random.n_rows, random.n_cols, false), 
    random_weights_(rweights),
    num_resampling_regions_(num_regions), data_box_length_(box_length),
    data_trees_(num_regions),
    data_mats_(num_regions), leaf_size_(leaf_size),
    data_weights_(num_regions),
    {
      
      tuple_size_ = 3;
      
      // I think I still need to do this
      // They will hopefully still exist outside, right?
      for (int i = 0; i < num_resampling_regions_; i++) {
        
        data_mats_[i] = new arma::mat;
        data_weights_[i] = new arma::colvec;
        
      } // for i
      
      // decide on the number of partitions
      int num_over_3 = num_resampling_regions_ / 3;
      
      num_height_partitions_ = num_over_3;
      num_width_partitions_ = num_over_3;
      num_depth_partitions_ = num_over_3;
      
      if (num_resampling_regions_ % 3 >= 1) {
        num_height_partitions_++;
      }
      if (num_resampling_regions_ % 3 == 2) {
        num_width_partitions_++;
      }
      
      // now, find the step sizes
      height_step_ = data_box_length_ / (double)num_height_partitions_;
      width_step_ = data_box_length_ / (double)num_width_partitions_;
      depth_step_ = data_box_length_ / (double)num_depth_partitions_;
      
      
      SplitData_();
      
      for (int i = 0; i < num_resampling_regions_; i++) {
        mlpack::IO::Info << "Region " << i <<": " << data_mats_[i].n_cols;
        mlpack::IO::Info << " points.\n";
      }
      
      
      BuildTrees_();
      
      
      
      
    } // constructor
    
    
    std::vector<arma::mat*>& data_mats() {
      return data_mats_;
    }
    
    // TODO: decide on pointers vs. references vs. whatever
    arma::mat& data_mat(int i) {
      return data_mats_[i];
    }

    std::vector<arma::colvec*>& data_weights() {
      return data_weights_;
    }
    
    arma::covec& data_weight(int i) {
      return data_weights_[i];
    }
    
    arma::mat& random_mat() {
      return random_mat_;
    }
    
    arma::colvec& random_weights() {
      return random_weights_;
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

