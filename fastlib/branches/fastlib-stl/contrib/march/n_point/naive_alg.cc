/*
 *  naive_alg.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "naive_alg.h"

void npt::NaiveAlg::ComputeCountsHelper_(std::vector<bool>& permutation_ok, 
                                         std::vector<index_t>& points_in_tuple,
                                         int k) {
 
  std::vector<bool> permutation_ok_copy(permutation_ok);
  
  // We don't have any points in the tuple, so try adding all of them
  if (k == 0) {
  
    for (index_t i = 0; i < num_points_; i++) {
    
      points_in_tuple[0] = i;
      ComputeCountsHelper_(permutation_ok, points_in_tuple, k+1);
      
    } // for i
  
  
  }
  else {
    // k > 0
    
    // all points with index <= k will violate symmetry in this tuple
    for (index_t new_point = points_in_tuple[k-1] + 1; new_point < num_points_; 
         new_point++) {
      
      bool this_point_works = true;
      arma::colvec new_point_vec = data_points_.col(new_point);
      
      // TODO: does this leak memory?
      permutation_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
      
      // loop over points already in the tuple
      for (index_t j = 0; this_point_works && j < k; j++) {
        
        index_t old_point = points_in_tuple[j];
        
        arma::colvec old_point_vec = data_points_.col(old_point);

        double point_dist_sq = la::DistanceSqEuclidean(new_point_vec, 
                                                       old_point_vec);
        
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k,
                                                  permutation_ok_copy);
        
      } // for j
      
      // did the point work?
      if (this_point_works) {
        
        points_in_tuple[k] = new_point;
        
        // are we finished?
        if (k == tuple_size_ - 1) {
          
          //std::cout << "valid tuple found\n";
          
          num_tuples_++;
          double this_weight = 1.0;
          
          for (index_t tuple_ind = 0; tuple_ind < tuple_size_; tuple_ind++) {
            
            this_weight *= data_weights_(points_in_tuple[tuple_ind]);
            
          } // tuple_ind
          
          weighted_num_tuples_ += this_weight;
          
        }
        else {
          
          ComputeCountsHelper_(permutation_ok_copy, points_in_tuple, k+1);
          
        }
        
      } // the point works
      
    } // for new_point
    
  } // k > 0
  
  
} // ComputeCountsHelper_()


void npt::NaiveAlg::ComputeCounts() {

  std::vector<bool> permutation_ok(matcher_.num_permutations(), true);
  
  std::vector<index_t> points_in_tuple(tuple_size_, -1);
  
  ComputeCountsHelper_(permutation_ok, points_in_tuple, 0);
  
} // ComputeCounts()

