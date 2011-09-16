/*
 *  single_results.cc
 *  
 *
 *  Created by William March on 9/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "single_results.h"

void npt::SingleResults::AddResult_(int region_ind, int num_random, 
                                    int r1_ind, int theta_ind, int result) {
  
  results_[region_ind][num_random][r1_ind][theta_ind] += result;
  
} // AddResult()

void npt::SingleResults::AddRandomResult_(int r1_ind, int theta_ind,
                                          int result) {
  
  RRR_result_[r1_ind][theta_ind] += result;
  
}


void npt::SingleResults::ProcessResults(int region_id, 
                                        int num_random,
                                        SingleMatcher& matcher,
                                        int r1_ind, int theta_ind) {
  
  if (num_random == tuple_size_ + 1) {
    
    AddRandomResult_(r1_ind, theta_ind, matcher.results());
    
  }
  else {
  
    for (int i = 0; i < num_regions_; i++) {
      
      if (region_id != i) {
        AddResult_(i, num_random, r1_ind, theta_ind, matcher.results());
      }
    } // for regions

  }
  
} // ProcessResults()


void npt::SingleResults::PrintResults() {

  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  mlpack::IO::Info << "Multi-angle Resampling Results: \n\n";
  
  for (int region_ind = 0; region_ind < num_regions_; region_ind++) {
    
    mlpack::IO::Info << "Resampling region " << region_ind << "\n";
    
    for (int num_random = 0; num_random < tuple_size_; num_random++) {
      
      std::string this_string(label_string, num_random, tuple_size_);
      mlpack::IO::Info << this_string << ": \n";
      
      for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
        
        for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
          
          mlpack::IO::Info << "r1: " << r1_vec_[r1_ind] << ", theta: ";
          mlpack::IO::Info << theta_vec_[theta_ind] << ": ";
          mlpack::IO::Info << results_[region_ind][num_random][r1_ind][theta_ind];
          mlpack::IO::Info << "\n";
          
        } // for theta
        
      } // for r1_ind
      
      mlpack::IO::Info << "\n";
      
    } // for num_random
    
  } // for region_ind
  
  mlpack::IO::Info << "\nRRR results: \n";
  
  for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
    
    for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
      
      mlpack::IO::Info << "r1: " << r1_vec_[r1_ind] << ", theta: ";
      mlpack::IO::Info << theta_vec_[theta_ind] << ": ";
      mlpack::IO::Info << RRR_result_[r1_ind][theta_ind];
      mlpack::IO::Info << "\n";
      
    } // for theta
    
  } // for r1_ind
  
  mlpack::IO::Info << "\n";
  
  
} // PrintREsults()