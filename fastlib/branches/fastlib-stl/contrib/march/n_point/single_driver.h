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

namespace npt {

  class SingleDriver {
    
  private:
    
    static int tuple_size_ = 3;
    
    SingleResults results_;
    
    NaiveResampling resampling_class_;
    
    std::vector<double> r1_vec_;
    double r2_multiplier_;
    std::vector<double> theta_vec_;
    double bin_size_;
    
    AngleMatcherGenerator generator_;
    
    
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
                 double box_x_length, double box_y_length, double box_z_length, 
                 int leaf_size) : 
    r1_vec_(short_sides), r2_multiplier_(long_side), theta_vec_(thetas),
    bin_size_(bin_size),
    resampling_class_(data_mat, data_weights, 
                      random_mat, random_weights, num_regions, box_length,
                      leaf_size),
    results_(num_regions, short_sides, thetas),
    generator_(r1_vec_, r2_multiplier_, theta_vec_, bin_size_)
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

