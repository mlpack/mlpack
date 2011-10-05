/*
 *  test_single_bandwidth.h
 *  
 *
 *  Created by William March on 10/4/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TEST_SINGLE_BANDWIDTH_H
#define TEST_SINGLE_BANDWIDTH_H

#include <boost/test/unit_test.hpp>
#include <mlpack/core.h>
#include "generic_npt_alg.h"
#include "single_matcher.h"

// include boost random number generation stuff
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>


namespace npt {

  class TestSingleBandwidth {
    
  private:
    
    boost::random::mt19937 generator_;
    boost::random::uniform_int_distribution<> num_data_dist; 
    boost::random::uniform_int_distribution<> tuple_size_dist;
    
    boost::random::uniform_int_distribution<> num_leaves_dist;
    
    boost::random::uniform_01<double> data_dist;
    boost::random::uniform_real_distribution<double> matcher_dist;
    boost::random::uniform_real_distribution<double> matcher_thick_dist;
    
    
    void GenerateRandomSet_(arma::mat& data);
    
    // fills the matcher with distances and returns the thickness
    double GenerateRandomMatcher_(arma::mat& matcher);
    
  public:
    
    TestSingleBandwidth() : 
    num_data_dist(500,1000), tuple_size_dist(2,4),
    num_leaves_dist(1,50), matcher_dist(0.05, 0.25), 
    matcher_thick_dist(0.05,0.1)
    {}
    
    bool StressTest();
    
    
    // the driver
    // needs to initialize the random stuff
    bool StressTestMain();
    
    
  }; // class
  
} // namespace 

#endif 
