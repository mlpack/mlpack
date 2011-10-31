/*
 *  generate_random_problem.cc
 *  
 *
 *  Created by William March on 10/31/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "generate_random_problem.h"

void npt::GenerateRandomProblem::GenerateRandomSet(arma::mat& data) {
  
  for (unsigned int row_ind = 0; row_ind < data.n_rows; row_ind++) {
    
    for (unsigned int col_ind = 0; col_ind < data.n_cols; col_ind++) {
      
      data(row_ind,col_ind) = data_gen();
      
    }
    
  }
  
} // GenerateRandomSet

// fills the matcher with distances and returns the thickness
double npt::GenerateRandomProblem::GenerateRandomMatcher(arma::mat& matcher) {
  
  for (unsigned int i = 0; i < matcher.n_rows; i++) {
    
    matcher(i,i) = 0.0;
    
    for (unsigned int j = i+1; j < matcher.n_cols; j++) {
      
      matcher(i,j) = matcher_dist_gen();
      matcher(j,i) = matcher(i,j);
      
    }
    
  }
  
  return matcher_thick_gen();
  
} 

void npt::GenerateRandomProblem::GenerateRandomMatcher(arma::mat& lower_bounds, 
                                                       arma::mat& upper_bounds) {
  
  assert(lower_bounds.n_cols == lower_bounds.n_rows);
  assert(lower_bounds.n_cols == upper_bounds.n_rows);
  assert(upper_bounds.n_rows == upper_bounds.n_cols);
  
  for (unsigned int i = 0; i < lower_bounds.n_rows; i++) {
    
    lower_bounds(i,i) = 0.0;
    upper_bounds(i,i) = 0.0;
    
    for (unsigned int j = i+1; j < lower_bounds.n_cols; j++) {
      
      double val1 = matcher_dist_gen();
      double val2 = matcher_dist_gen();
      
      upper_bounds(i,j) = std::max(val1, val2);
      upper_bounds(j,i) = upper_bounds(i,j);
      
      lower_bounds(i,j) = std::min(val1, val2);
      lower_bounds(j,i) = lower_bounds(i,j);
      
    }
    
  }
  
} // GenerateRandomMatcher (upper and lower)
