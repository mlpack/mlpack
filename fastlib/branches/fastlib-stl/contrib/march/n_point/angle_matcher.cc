/*
 *  angle_matcher.cc
 *  
 *
 *  Created by William March on 7/26/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "angle_matcher.h"



// given two edges and the angle between them, compute the length of the 
// third size
// TODO: keep things squared?
double npt::AngleMatcher::ComputeR3_(double r1, double r2, double theta) {
  
  double r3sqr = (r1 * r1) + (r2 * r2) - 2.0 * r1 * r2 * math::cos(theta);

  return math::sqrt(r3sqr);
  
} 




// returns the index of the value of r1 that is satisfied by the tuple
// the list contains the indices of thetas_ that are satisfied by the tuple
// assumes that valid_theta_indices is initialized and empty
// Important: it is possible to have a tuple satisfy more than one matcher
int npt::AngleMatcher::TestPointTuple(arma::colvec& vec1, arma::colvec& vec2, 
                                      arma::colvec& vec3,
                                      std::vector<int>& valid_theta_indices) {
  
  // TODO: profile this while optimized 
  
  double d12_sqr = la::DistanceSqEuclidean(vec1, vec2);
  double d13_sqr = la::DistanceSqEuclidean(vec1, vec3);
  double d23_sqr = la::DistanceSqEuclidean(vec3, vec2);
  
  std::vector<double> sorted_dists_sq(3);
  sorted_dists_sq[0] = d12_sqr;
  sorted_dists_sq[1] = d13_sqr;
  sorted_dists_sq[2] = d23_sqr;
  
  std::sort(sorted_dists_sq.begin(), sorted_dists_sq.end());

  // IMPORTANT: r3 upper and lower may not be strictly sorted, especially for 
  // the larger ones
  
  
  int r1_index = -1;
  // Find the correct value of r1
  // IMPORTANT: assuming that only one value of r1 is satisfied
  for (int i = 0; i < r1_lower_sqr_.size(); i++) {
    
    if ((sorted_dists_sq[0] > r1_lower_sqr_[i]) 
        && (sorted_dists_sq[0] < r1_upper_sqr_[i])) {
      r1_index = i;
      break;
    }
    
  }
  
  // IMPORTANT: I'm not sure this works right at the transition point, i.e.
  // d2 might be less than d1, but they fit in the other order
  
  // make sure it fits some r1
  if (r1_index >= 0) {
    
    // do the ones where r3 < r2
    for (int r3_index = 0; r3_index < theta_cutoff_index_; r3_index++) {
      
      if ((sorted_dists_sq[2] > r2_lower_sqr_[r1_index])
          && (sorted_dists_sq[2] < r2_upper_sqr_[r1_index])
          && (sorted_dists_sq[1] > r3_lower_sqr_[r3_index])
          && (sorted_dists_sq[1] < r3_upper_sqr_[r3_index])) {
      
        // add r3 index to the list of valid stuff
        valid_theta_indices.push_back(r3_index);
        
      }
      
    } // r3_index

    // now, r3 >= r2
    for (int r3_index = theta_cutoff_index_; r3_index < thetas_.size(); r3_index++) {
      
      if ((sorted_dists_sq[1] > r2_lower_sqr_[r1_index])
          && (sorted_dists_sq[1] < r2_upper_sqr_[r1_index])
          && (sorted_dists_sq[2] > r3_lower_sqr_[r3_index])
          && (sorted_dists_sq[2] < r3_upper_sqr_[r3_index])) {
        
        // add r3 index to the list of valid stuff
        valid_theta_indices.push_back(r3_index);
        
      }
      
    } // r3_index

    // we need this in case we found a valid r1 but no valid r2/r3 combos
    if (valid_theta_indices.size() == 0) {
      r1_index = -1;
    }
    
  } // find the valid thetas
  
  return r1_index;
  
} // TestPointTuple

// returns true if the tuple of nodes might contain a tuple of points that
// satisfy one of the matchers
// If false, then pruning is ok
bool npt::AngleMatcher::TestNodeTuple(const DHrectBound<2>& box1, 
                                      const DHrectBound<2>& box2,
                                      const DHrectBound<2>& box3) {
  
  // pruning options: all three distances are shorter than the shortest r1
  // or longer than the longest one
  
  double d12_lower = box1.MinDistanceSq(box2);
  double d12_upper = box1.MaxDistanceSq(box2);

  double d13_lower = box1.MinDistanceSq(box3);
  double d13_upper = box1.MaxDistanceSq(box3);

  double d23_lower = box2.MinDistanceSq(box3);
  double d23_upper = box2.MaxDistanceSq(box3);

  // find the valid r1's - the smallest upper bound needs to fit
  // check that there is a valid r2/r3 for each
  
  
  
  
} // TestNodeTuple

