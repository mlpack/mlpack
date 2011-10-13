/*
 *  jackknife_resampling.h
 *  
 *
 *  Created by William March on 8/24/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef JACKKNIFE_RESAMPLING_H
#define JACKKNIFE_RESAMPLING_H

#include "fastlib/fastlib.h"
#include "matcher_generation.h"
#include "generic_npt_alg.h"

// for now, only writing this for the angle version


namespace npt {
  
  // TODO: figure out how to permute the weights after building trees
  
  class JackknifeResampling {
    
  private:
    
    arma::mat data_all_mat_;
    arma::colvec data_all_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    
    NptNode* random_tree_;
    
    std::vector<NptNode*> data_trees_;
    
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
    
    // indexed by [resampling_region][num_random][r1][theta]
    boost::multi_array<int, 4> results_;
    boost::multi_array<double, 4> weighted_results_;
    
    // matcher info
    std::vector<double> r1_vec_;
    double r2_mult_;
    std::vector<double> theta_vec_;
    double bin_width_;

    
    ///////////////// functions ////////////////////////
    int FindRegion_(arma::colvec& col);

    
    void SplitData_();
    
    void BuildTrees_();
    
    void ProcessResults_(boost::multi_array<double, 2>& this_result,
                         arma::col& this_region,
                         int num_random);
      
    
  public:
    
    JackknifeResampling(arma::mat& data, arma::colvec& weights,
                        arma::mat& random, arma::colvec& rweights,
                        int num_regions, double box_length,
                        std::vector<double>& r1, double r2_mult,
                        std::vector<double>& thetas, double bin_width,
                        int leaf_size) :
    // trying to avoid copying data
    data_all_mat_(data.memptr(), data.n_rows, data.n_cols, false), 
    data_all_weights_(weights),
    random_mat_(random.memptr(), random.n_rows, random.n_cols, false), 
    random_weights_(rweights),
    num_resampling_regions_(num_regions), data_box_length_(box_length),
    data_trees_(num_regions),
    results_(boost::extents[num_regions][4][r1.size()][thetas.size()],
             c_storage_order(), 0),
    weighted_results_(boost::extents[num_regions][4][r1.size()][thetas.size()],
                      c_storage_order(), 0.0),
    data_mats_(num_regions), leaf_size_(leaf_size),
    data_weights_(num_regions),
    r1_vec_(r1), r2_mult_(r2_mult), theta_vec_(thetas), bin_width_(bin_width)
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
        mlpack::Log::Info << "Region " << i <<": " << data_mats_[i].n_cols << "\n";
      }
      
      
      BuildTrees_();
      
      
      
      
    } // constructor
    
    
    void Compute();
    
    void PrintResults();
    
    
    
  }; // class
  
  
} // namespace




#endif

