/*
 *  angle_3pt_alg.cc
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "angle_results.h"

void npt::AngleResults::AddResult_(int region_id, int num_random, 
                                   boost::multi_array<int, 2>& partial_result) {
  
  for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
    
    for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
      
      results_[region_id][num_random][r1_ind][theta_ind] 
              += partial_result[r1_ind][theta_ind];
      
    } // for theta
    
  } // for r1
  
} // AddResult_

void npt::AngleResults::AddRandomResult_(boost::multi_array<int, 2>& partial_result) {
  
  for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
    
    for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
      
      RRR_result_[r1_ind][theta_ind] += partial_result[r1_ind][theta_ind];
      
    } // for theta
    
  } // for r1
  
} // AddRandomResult


void npt::AngleResults::ProcessResults(std::vector<int>& region_ids, 
                                       int num_random,
                                       AngleMatcher& matcher) {
  
  if (num_random == tuple_size_) {
    
    AddRandomResult_(matcher.results());
    
  }
  else {

    for (int i = 0; i < num_regions_; i++) {

      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
      } // check the invalid region ids
      
      if (!skip_me) {
        AddResult_(i, num_random, matcher.results());
      }
    } // for i
    
  }
    
} // Process Results


void npt::AngleResults::PrintResults() {
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  mlpack::Log::Info << "Multi-angle Resampling Results: \n\n";
  
  for (int region_ind = 0; region_ind < num_regions_; region_ind++) {
    
    mlpack::Log::Info << "Resampling region " << region_ind << "\n";
    
    for (int num_random = 0; num_random < tuple_size_; num_random++) {
      
      std::string this_string(label_string, num_random, tuple_size_);
      mlpack::Log::Info << this_string << ": \n";
      
      for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
        
        for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
          
          mlpack::Log::Info << "r1: " << r1_vec_[r1_ind] << ", theta: ";
          mlpack::Log::Info << theta_vec_[theta_ind] << ": ";
          mlpack::Log::Info << results_[region_ind][num_random][r1_ind][theta_ind];
          mlpack::Log::Info << "\n";
          
        } // for theta
        
      } // for r1_ind
      
      mlpack::Log::Info << "\n";
      
    } // for num_random
    
  } // for region_ind
  
  mlpack::Log::Info << "\nRRR results: \n";
  
  for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
    
    for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
      
      mlpack::Log::Info << "r1: " << r1_vec_[r1_ind] << ", theta: ";
      mlpack::Log::Info << theta_vec_[theta_ind] << ": ";
      mlpack::Log::Info << RRR_result_[r1_ind][theta_ind];
      mlpack::Log::Info << "\n";
      
    } // for theta
    
  } // for r1_ind
  
  mlpack::Log::Info << "\n";
  
  
} // PrintResults
