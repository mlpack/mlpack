/*
 *  generate_random_problem.h
 *  
 *
 *  Created by William March on 10/31/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GENERATE_RANDOM_PROBLEM_H
#define GENERATE_RANDOM_PROBLEM_H

//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_int_distribution.hpp>
//#include <boost/random/uniform_real_distribution.hpp>
//#include <boost/random/uniform_01.hpp>
//#include <boost/random/variate_generator.hpp>
#include <boost/random.hpp>

#include <mlpack/core.h>



namespace npt {

  class GenerateRandomProblem {
    
  private:
    
    boost::random::mt19937 generator_;
    boost::random::uniform_int_distribution<> num_data_dist; 
    
    boost::random::uniform_01<double> data_dist;
    boost::random::uniform_real_distribution<double> matcher_dist;
    boost::random::uniform_real_distribution<double> matcher_thick_dist;
    
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_01<> > data_gen;
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_real_distribution<> > matcher_dist_gen;
    boost::random::variate_generator<boost::random::mt19937, 
    boost::random::uniform_real_distribution<> > matcher_thick_gen;
    
    
  public:
    
    void GenerateRandomSet(arma::mat& data);
    
    // fills the matcher with distances and returns the thickness
    double GenerateRandomMatcher(arma::mat& matcher);
    
    void GenerateRandomMatcher(arma::mat& lower_bounds, 
                               arma::mat& upper_bounds);

    
    GenerateRandomProblem() : 
    generator_(std::time(0)),
    matcher_dist(0.05, 0.25), 
    matcher_thick_dist(0.05,0.1),
    data_gen(generator_, data_dist),
    matcher_dist_gen(generator_, matcher_dist),
    matcher_thick_gen(generator_, matcher_thick_dist)
    {
      //printf("constructed test class\n");
    }
    
    
    
  }; // class
  
} 


#endif