/*
 *  angle_matcher_generation.h
 *  
 *
 *  Created by William March on 8/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ANGLE_MATCHER_GENERATION_H 
#define ANGLE_MATCHER_GENERATION_H

#include <mlpack/core.h>

namespace npt {
  
  class AngleMatcherGenerator {
    
  private:
    
    std::vector<arma::mat> lower_bound_vec_;
    std::vector<arma::mat> upper_bound_vec_;
    
    int num_r1_;
    int num_theta_;
    
    
    // computes these (upper and lower bound vecs) and adds them to the end of 
    // the arrays
    void ComputeMatcher_(double r1, double r2, double theta, double bin_width);
    
    
    
  public:
    
    
    AngleMatcherGenerator(std::vector<double>& r1_vec, double r2_mult,
                          std::vector<double> theta_vec, double bin_width) 
    {
      
      num_r1_ = r1_vec.size();
      num_theta_ = theta_vec.size();
      
      for (unsigned int r1_ind = 0; r1_ind < r1_vec.size(); r1_ind++) {
        
        for (unsigned int theta_ind = 0; theta_ind < theta_vec.size();
             theta_ind++) {
          
          ComputeMatcher_(r1_vec[r1_ind], r1_vec[r1_ind] * r2_mult, 
                          theta_vec[theta_ind], bin_width);
          
        } // for thetas
        
      } // for r1
      
      
    } // constructor
    
    arma::mat& lower_matcher(int r1_ind, int theta_ind) {

      int i = r1_ind * num_theta_ + theta_ind;
      
      return lower_bound_vec_[i];

    }

    arma::mat& upper_matcher(int r1_ind, int theta_ind) {
      
      int i = r1_ind * num_theta_ + theta_ind;
      
      return upper_bound_vec_[i];
      
    }
    
    int num_matchers() {
      return lower_bound_vec_.size();
    }
    
  };  // class
  
  
  
} // namespace



#endif