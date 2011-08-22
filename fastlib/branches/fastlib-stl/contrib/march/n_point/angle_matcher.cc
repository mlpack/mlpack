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
  
  double r3sqr = (r1 * r1) + (r2 * r2) - 2.0 * r1 * r2 * cos(theta);

  return sqrt(r3sqr);
  
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
          && (sorted_dists_sq[1] > r3_lower_sqr_[r1_index][r3_index])
          && (sorted_dists_sq[1] < r3_upper_sqr_[r1_index][r3_index])) {
      
        // add r3 index to the list of valid stuff
        valid_theta_indices.push_back(r3_index);
        
      }
      
    } // r3_index

    // now, r3 >= r2
    for (int r3_index = theta_cutoff_index_; r3_index < thetas_.size(); r3_index++) {
      
      if ((sorted_dists_sq[1] > r2_lower_sqr_[r1_index])
          && (sorted_dists_sq[1] < r2_upper_sqr_[r1_index])
          && (sorted_dists_sq[2] > r3_lower_sqr_[r1_index][r3_index])
          && (sorted_dists_sq[2] < r3_upper_sqr_[r1_index][r3_index])) {
        
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
/*
bool npt::AngleMatcher::TestNodeTuple(const DHrectBound<2>& box1, 
                                      const DHrectBound<2>& box2,
                                      const DHrectBound<2>& box3) {
  
  // pruning options: all three distances are shorter than the shortest r1
  // or longer than the longest one
  
  bool possibly_valid = true;
  
  double d12_lower = box1.MinDistanceSq(box2);
  double d12_upper = box1.MaxDistanceSq(box2);

  double d13_lower = box1.MinDistanceSq(box3);
  double d13_upper = box1.MaxDistanceSq(box3);

  double d23_lower = box2.MinDistanceSq(box3);
  double d23_upper = box2.MaxDistanceSq(box3);

  // find the valid r1's - the smallest upper bound needs to fit
  // check that there is a valid r2/r3 for each
  
  std::vector<double> sorted_upper_sq(3);
  std::vector<double> sorted_lower_sq(3);
  
  sorted_upper_sq[0] = d12_upper;
  sorted_upper_sq[1] = d13_upper;
  sorted_upper_sq[2] = d23_upper;
  
  sorted_lower_sq[0] = d12_lower;
  sorted_lower_sq[1] = d13_lower;
  sorted_lower_sq[2] = d23_lower;
  
  std::sort(sorted_upper_sq.begin(), sorted_upper_sq.end());
  std::sort(sorted_lower_sq.begin(), sorted_lower_sq.end());

  // is it possible for the smallest and largest to be okay, but the middle one 
  // is not?
  
  // if the smallest lower bound is larger than the largest value of r1, we prune
  // we can also prune if the smallest upper bound is too small to be r1
  
  // this is an r1 prune
  if (sorted_lower_sq[0] > r1_upper_sqr_.back()
      || sorted_upper_sq[0] < r1_lower_sqr_.front()) {
    possibly_valid = false;
    num_r1_prunes_++;
  }
  // we can also prune if the largest values don't fit r3
  // Not sure if this is correct - using middle upper value
  // this is an r3 prune
  else if (false) {
  //else if (sorted_lower_sq[2] > r3_upper_sqr_.back().back()
  //         || (sorted_upper_sq[1] < r3_lower_sqr_.front().front())) {
    possibly_valid = false;
    num_r3_prunes_++;
  }
   
  // we can prune if r2 doesn't fit either
  // i.e. the smallest possible r2
  // Problem: can't assume that the middle value of sorted_lower_sq_ goes with
  // r2 -- it might go with r3
  
  else if (sorted_lower_sq_[1] > r2_upper_sq_[2]) {
    
  }
  else {
    
    double this_min_perimeter = (sqrt(d12_lower) 
                                 + sqrt(d13_lower) 
                                 + sqrt(d23_lower)) / 2.0;
    double this_max_perimeter = (sqrt(d12_upper) 
                                 + sqrt(d13_upper) 
                                 + sqrt(d23_upper)) / 2.0;
    
    double this_max_area_sq;
    double this_min_area_sq;
  
    this_min_area_sq = this_min_perimeter 
    * (this_min_perimeter - sqrt(d12_lower)) 
    * (this_min_perimeter - sqrt(d13_lower)) 
    * (this_min_perimeter - sqrt(d23_lower));
    this_max_area_sq = this_max_perimeter 
    * (this_max_perimeter - sqrt(d12_upper)) 
    * (this_max_perimeter - sqrt(d13_upper)) 
    * (this_max_perimeter - sqrt(d23_upper));
    
    if (this_min_area_sq > max_area_sq_) {
      num_min_area_prunes_++;
      possibly_valid = false;
    }
    else if (this_max_area_sq < min_area_sq_) {
      num_max_area_prunes_++;
      possibly_valid = false;
    }
    
  } // area pruning
  
  
  // Want: another prune that takes into account the constraint on r2 once 
  // we've chosen an r1
  
  return possibly_valid;
  
} // TestNodeTuple
*/

// Trying this again
bool npt::AngleMatcher::TestNodeTuple(const DHrectBound<2>& box1, 
                                      const DHrectBound<2>& box2,
                                      const DHrectBound<2>& box3) {
  
  // pruning options: all three distances are shorter than the shortest r1
  // or longer than the longest one
  
  bool possibly_valid = true;
  
  double d12_lower = box1.MinDistanceSq(box2);
  double d12_upper = box1.MaxDistanceSq(box2);
  
  double d13_lower = box1.MinDistanceSq(box3);
  double d13_upper = box1.MaxDistanceSq(box3);
  
  double d23_lower = box2.MinDistanceSq(box3);
  double d23_upper = box2.MaxDistanceSq(box3);
  
  // find the valid r1's - the smallest upper bound needs to fit
  // check that there is a valid r2/r3 for each
  
  std::vector<double> sorted_upper_sq(3);
  std::vector<double> sorted_lower_sq(3);
  
  sorted_upper_sq[0] = d12_upper;
  sorted_upper_sq[1] = d13_upper;
  sorted_upper_sq[2] = d23_upper;
  
  sorted_lower_sq[0] = d12_lower;
  sorted_lower_sq[1] = d13_lower;
  sorted_lower_sq[2] = d23_lower;
  
  std::sort(sorted_upper_sq.begin(), sorted_upper_sq.end());
  std::sort(sorted_lower_sq.begin(), sorted_lower_sq.end());

  /////////
  
  // r1 is always the smallest part of the triangle
  // r2 or r3 may be smaller, depending on whether we are using a value of 
  // theta larger or smaller than 
  
  // IMPORTANT: remember that r1, r2, r3 are fixed, the permutation takes place
  // in the sorting step above (i.e. in l1, l2, l3; u1, u2, u3)
  
  // if the smallest possible pairwise distance is larger than the largest 
  // possible value of r1, we can prune
  //if (false) {
  if (sorted_lower_sq.front() > r1_upper_sqr_.back()) {
    possibly_valid = false;
    num_large_r1_prunes_++;
  }
  // if the largest possible value of r1
  else if (sorted_upper_sq.front() < shortest_possible_side_sqr_) {
  //else if (false) {
    possibly_valid = false;
    num_small_r1_prunes_++;
  }
  // can check if the largest values don't fit too
  // IMPORTANT: this assumes the largest value of theta is greater than theta*
  // if not, then the largest r2 value is truly the largest
  // set this in the constructor
  //else if (false) {
  else if (sorted_lower_sq.back() > longest_possible_side_sqr_) {
    possibly_valid = false;
    num_large_r3_prunes_++;
  }
  // I could check for the longest pairwise distance being too short, but I 
  // don't think that will happen very often
  // check the structure of the triangle
  else {
    
    //bool small_angle_valid = true;
    //bool large_angle valid = true;
    
    double largest_r2, smallest_r2;
    //double largest_r3, smallest_r3;
    
    // TODO: do I need to make these correspond more to the actual values?
    
    // check large angles (theta >= theta*)
  
    // compute largest and smallest r2
    // TODO: is this correct?
    largest_r2 = long_side_multiplier_ * long_side_multiplier_ * sorted_upper_sq.front();
    smallest_r2 = long_side_multiplier_ * long_side_multiplier_ * sorted_lower_sq.front();
    
    if (false) {
    //if (sorted_lower_sq[1] > largest_r2) {
      //large_angle_valid = false;
      possibly_valid = false;
      num_large_r2_prunes_++;
    }
    else if (false) {
    //else if (sorted_upper_sq[1] < smallest_r2) {
      //large_angle_valid = false;
      possibly_valid = false;
      num_small_r2_prunes_++;
    }

    // IMPORTANT: do I need to check this? 
    
    // The three distances (l1, l2, l3) etc. came from a real triangle
    // so the third distance (up to differences in bounds) satisfies the 
    // triangle inequality already
    
    // The only way to prune based on r3 is if the 'true' angle is too wide or
    // too steep
    // However, inputs tend to favor extremely large or small angles  
    
    // if we haven't pruned large angles, then compute and check r3
    
    /*
    if (large_angle_valid) {
      
      // compute largest and smallest r3
      
      if (sorted_lower_sq.back() > largest_r3) {
        large_angle_valid = false;
      }
      else if (sorted_upper_sq.back() < smallest_r3) {
        large_angle_valid = false;
      }
    } // checking large angles
     */  
    
    // check small angles

    // I don't think I actually need two cases if I'm not worrying about r3
    
    /*
    // recompute largest and smallest r3
    if (sorted_lower_sq[1] > largest_r3) {
      small_angle_valid = false;
    }
    else if (sorted_upper_sq[1] < smallest_r3) {
      small_angle_valid = false;
    }
    
    // if we havent pruned, check the third side
    if (small_angle_valid) {
     
      // recompute smallest and largest r2
      
      if (sorted_lower_sq.back() > largest_r2) {
        small_angle_valid = false;
      }
      else if (sorted_uper_sq.back() < smallest_r2) {
        small_angle_valid = false;
      }
      
    } // checking small angles
    
    // if either might work, we can' prune
    possibly_valid = (small_angle_valid || large_angle_valid);
    */
    
  } // cant prune on r1
  
  return possibly_valid;
  
} // TestNodeTuple




