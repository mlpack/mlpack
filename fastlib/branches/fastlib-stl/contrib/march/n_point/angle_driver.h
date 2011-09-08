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

namespace npt {

  class AngleDriver {
    
  private:
    
    static int tuple_size_ = 3;
    
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
                int num_regions, double box_length, int leaf_size) :
    r1_vec_(short_sides), r2_multiplier_(long_side), theta_vec_(thetas),
    bin_size_(bin_size),
    resampling_class_(data_mat, data_weights, 
                      random_mat, random_weights, num_regions, box_length,
                      leaf_size)
    results_(num_regions, short_sides, thetas)
    { } // constructor
    
    
    void Compute();
      
    
    void PrintResults();
    
    
    
  }; // class

} //namespace


#endif