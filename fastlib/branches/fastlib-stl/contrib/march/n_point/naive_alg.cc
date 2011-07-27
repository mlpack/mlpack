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

  bool adding_random = (k < num_random_);

  int new_point_ind;
  int num_new_points;
  int previous_point_ind;

  if (adding_random) {
    num_new_points = num_random_points_;
  }
  else {
    num_new_points = num_points_;
  }
  
  // we need to start adding from the beginning of the list here
  if (k == 0 || k == num_random_) {
    previous_point_ind = -1;
  }
  else {
    previous_point_ind = points_in_tuple[k-1];
  }
  
  /////////////////////////////////////////////
  // all points with index <= k will violate symmetry in this tuple

  // need to make sure that I take the new point from the correct set 
  // and check the symmetry correctly
    for (new_point_ind = previous_point_ind + 1; new_point_ind < num_new_points; 
         new_point_ind++) {
      
      bool this_point_works = true;
      
      arma::colvec new_point_vec;
      if (adding_random) {
        new_point_vec = random_points_.col(new_point_ind);
      }
      else {
        new_point_vec = data_points_.col(new_point_ind);
      }
      // TODO: does this leak memory?
      permutation_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
      
      // loop over points already in the tuple
      for (index_t j = 0; this_point_works && j < k; j++) {
        
        index_t old_point = points_in_tuple[j];
        
        arma::colvec old_point_vec;
        if (j < num_random_) {
          old_point_vec = random_points_.col(old_point);
        }
        else {
          old_point_vec = data_points_.col(old_point);
        }
        
        double point_dist_sq = la::DistanceSqEuclidean(new_point_vec, 
                                                       old_point_vec);
        
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k,
                                                  permutation_ok_copy);
        
      } // for j
      
      // did the point work?
      if (this_point_works) {
        
        points_in_tuple[k] = new_point_ind;
        
        // are we finished?
        if (k == tuple_size_ - 1) {
  
          num_tuples_[num_random_]++;
          double this_weight = 1.0;
          
          for (index_t tuple_ind = 0; tuple_ind < num_random_; tuple_ind++) {
            
            this_weight *= random_weights_(points_in_tuple[tuple_ind]);
            
          } // tuple_ind
          
          for (int tuple_ind = num_random_; tuple_ind < tuple_size_; tuple_ind++) {
           
            this_weight *= data_weights_(points_in_tuple[tuple_ind]);
            
          }
          
          weighted_num_tuples_[num_random_] += this_weight;
          
        }
        else {
          
          ComputeCountsHelper_(permutation_ok_copy, points_in_tuple, k+1);
          
        }
        
      } // the point works
      
    } // for new_point
    
 
  
  
} // ComputeCountsHelper_()


void npt::NaiveAlg::ComputeCounts() {

  for (num_random_ = 0; num_random_ <= tuple_size_; num_random_++) {
    
    std::vector<bool> permutation_ok(matcher_.num_permutations(), true);
    
    std::vector<index_t> points_in_tuple(tuple_size_, -1);
    
    ComputeCountsHelper_(permutation_ok, points_in_tuple, 0);
        
  } // for num_random
  
} // ComputeCounts()

