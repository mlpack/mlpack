/*
 *  test_angle_resampling.h
 *  
 *
 *  Created by William March on 10/4/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TEST_ANGLE_RESAMPLING_H
#define TEST_ANGLE_RESAMPLING_H

#include "mlpack/core.h"
#include "angle_driver.h"
#include "single_driver.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>


namespace npt {

  class TestAngleResampling {
    
  private:
    
    boost::random::mt19937 generator_;
    
    boost::random::uniform_int_distribution<> num_data_dist; 
    
    boost::random::uniform_int_distribution<> num_leaves_dist;
    
    boost::random::uniform_01<double> data_dist;
    
    boost::random::uniform_int_distribution<> num_r1_dist;
    
    
    boost::random::uniform_int_distribution<> num_theta_dist;
    
    boost::random::uniform_real_distribution<double> r2_mult_dist;
    
    boost::random::uniform_real_distribution<double> bin_thick_dist;

    boost::random::uniform_int_distribution<> num_regions_dist;
    
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_int_distribution<> > num_data_gen;
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_int_distribution<> > num_leaves_gen;
    
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_01<> > data_gen;
    
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_int_distribution<> > num_r1_gen;

    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_int_distribution<> > num_theta_gen;
    
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_real_distribution<double> > r2_mult_gen;
    
    
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_real_distribution<double> > bin_thick_gen;
    
    
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_int_distribution<> > num_regions_gen;
    
    
    void GenerateRandomSet_(arma::mat& data);
    
    
  public:
    
    TestAngleResampling() : 
    generator_(std::time(0)),
    num_data_dist(100,300), 
    num_leaves_dist(1,25), 
    num_r1_dist(1, 3),
    num_theta_dist(1, 5),
    r2_mult_dist(1, 2.1),
    bin_thick_dist(0.05, 0.25),
    num_regions_dist(1, 3),
    num_data_gen(generator_, num_data_dist),
    num_leaves_gen(generator_, num_leaves_dist),
    data_gen(generator_, data_dist),
    num_r1_gen(generator_, num_r1_dist),
    num_theta_gen(generator_, num_theta_dist),
    r2_mult_gen(generator_, r2_mult_dist),
    bin_thick_gen(generator_, bin_thick_dist),
    num_regions_gen(generator_, num_regions_dist)    
    {
      printf("constructed test class\n");
    }
    
    bool StressTest();
    
    
    // the driver
    // needs to initialize the random stuff
    bool StressTestMain();
    
    
    
  }; // class
  
  
} // namespace

#endif 

