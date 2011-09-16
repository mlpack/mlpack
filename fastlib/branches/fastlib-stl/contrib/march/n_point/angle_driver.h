/*
 *  angle_3pt_alg.h
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ANGLE_DRIVER_H
#define ANGLE_DRIVER_H

#include "angle_matcher.h"
#include "angle_results.h"
#include "efficient_resampling.h"
#include "node_tuple.h"
#include "generic_npt_alg.h"

namespace npt {

  class AngleDriver {
    
  private:
    
    const static int tuple_size_ = 3;
    
    int num_resampling_regions_;
    
    AngleResults results_;
    
    EfficientResampling resampling_class_;
    
    std::vector<double> r1_vec_;
    double r2_multiplier_;
    std::vector<double> theta_vec_;
    double bin_size_;
    
        
    ///////////////////////////////
    
    
  public:
    
    AngleDriver(arma::mat& data_mat, arma::colvec& data_weights,
                arma::mat& random_mat, arma::colvec& random_weights,
                std::vector<double>& short_sides, double long_side,
                std::vector<double>& thetas, double bin_size,
                int num_x_regions, int num_y_regions, int num_z_regions,
                double box_x_length, double box_y_length, double box_z_length) :
    num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions),
    results_(num_resampling_regions_, short_sides, 
             thetas),
    resampling_class_(data_mat, data_weights, 
                      random_mat, random_weights, num_x_regions, num_y_regions,
                      num_z_regions, box_x_length, box_y_length, box_z_length),
    r1_vec_(short_sides), r2_multiplier_(long_side), theta_vec_(thetas),
    bin_size_(bin_size)
    { } // constructor
    
    
    void Compute();
      
    
    void PrintResults();
    
    
    
  }; // class

} //namespace


#endif