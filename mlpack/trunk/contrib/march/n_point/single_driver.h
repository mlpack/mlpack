/*
 *  single_bandwidth_alg.h
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 *
 *  Contains the single-bandwidth implementation algorithm class.
 *
 *  This is basically the same as the old auton code.
 */

#ifndef SINGLE_DRIVER_H
#define SINGLE_DRIVER_H

#include "single_matcher.h"
#include "single_results.h"
#include "naive_resampling.h"
#include "node_tuple.h"
#include "generic_npt_alg.h"
#include "angle_matcher_generation.h"

namespace npt {

  class SingleDriver {
    
  private:
    
    const static int tuple_size_ = 3;
    
    SingleResults results_;
    
    NaiveResampling resampling_class_;
    
    std::vector<double> r1_vec_;
    double r2_multiplier_;
    std::vector<double> theta_vec_;
    double bin_size_;
    
    AngleMatcherGenerator generator_;
    
    int num_resampling_regions_;
    
    
    ////////////////////// functions /////////////////////////
    
    
    
  public:
    
    /**
     * Requires the matcher bounds, data, parameters
     */
    SingleDriver(arma::mat& data, arma::colvec weights,
                 arma::mat& random, arma::colvec rweights,
                 std::vector<double>& short_sides, double long_side,
                 std::vector<double>& thetas, double bin_size,
                 int num_x_regions, int num_y_regions, int num_z_regions,
                 double box_x_length, double box_y_length, double box_z_length) : 
    results_((num_x_regions * num_y_regions * num_z_regions), 
             short_sides, thetas),
    resampling_class_(data, weights, 
                      random, rweights, 
                      num_x_regions, num_y_regions, num_z_regions, 
                      box_x_length, box_y_length, box_z_length),
    r1_vec_(short_sides), r2_multiplier_(long_side), theta_vec_(thetas),
    bin_size_(bin_size),
    generator_(r1_vec_, r2_multiplier_, theta_vec_, bin_size_),
    num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions)
    {
      
      
      
    } // constructor
    
    
    /**
     * Actually run the algorithm.
     */
    void Compute();
    
    void PrintResults();
    
  }; // class
  
  
  
} // namespace

#endif

