/*
 *  angle_matcher_generation.cc
 *  
 *
 *  Created by William March on 8/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "angle_matcher_generation.h"



void npt::AngleMatcherGenerator::ComputeMatcher_(double r1, double r2, 
                                                  double theta, 
                                                  double bin_width) {
  arma::mat lower_bounds(3,3);
  arma::mat upper_bounds(3,3);
  
  double half_width = bin_width / 2.0;
  
  double r3 = sqrt(r1*r1 + r2*r2 - 2.0*r1*r2*cos(theta));
  
  lower_bounds(0,0) = 0.0;
  lower_bounds(1,1) = 0.0;
  lower_bounds(2,2) = 0.0;
  
  upper_bounds(0,0) = 0.0;
  upper_bounds(1,1) = 0.0;
  upper_bounds(2,2) = 0.0;
  
  lower_bounds(0,1) = (1.0 - half_width) * r1;
  lower_bounds(1,0) = lower_bounds(0,1);
  lower_bounds(0,2) = (1.0 - half_width) * r2;
  lower_bounds(2,0) = lower_bounds(0,2);
  lower_bounds(1,2) = (1.0 - half_width) * r3;
  lower_bounds(2,1) = lower_bounds(1,2);
  
  
  
  upper_bounds(0,1) = (1.0 + half_width) * r1;
  upper_bounds(1,0) = upper_bounds(0,1);
  upper_bounds(0,2) = (1.0 + half_width) * r2;
  upper_bounds(2,0) = upper_bounds(0,2);
  upper_bounds(1,2) = (1.0 + half_width) * r3;
  upper_bounds(2,1) = upper_bounds(1,2);
  
  
  lower_bound_vec_.push_back(lower_bounds);
  upper_bound_vec_.push_back(upper_bounds);
  
} // ComputeMatcher_

