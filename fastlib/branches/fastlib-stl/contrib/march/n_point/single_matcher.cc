/*
 *  single_matcher.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "single_matcher.h"

bool npt::SingleMatcher::CheckDistances_(double dist_sq, index_t ind1, 
                                         index_t ind2) {
  
  return (dist_sq <= upper_bounds_sqr_(ind1, ind2) && 
          dist_sq >= lower_bounds_sqr_(ind1, ind2));
  
} //CheckDistances_


// note that this assumes that the points have been checked for symmetry
bool npt::SingleMatcher::TestPointPair(double dist_sq, index_t tuple_ind_1, 
                                       index_t tuple_ind_2,
                                       std::vector<bool>& permutation_ok) {

  bool any_matches = false;
  
  // iterate over all the permutations
  for (index_t i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    index_t template_index_1 = GetPermIndex_(i, tuple_ind_1);
    index_t template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
    //std::cout << "template_indices_checked\n";
    
    // Do the distances work?
    if (CheckDistances_(dist_sq, template_index_1, template_index_2)) {
      any_matches = true;
    }
    else {
      permutation_ok[i] = false;
    }
    
    // IMPORTANT: we can't exit here if any_matches is true
    // This is because the ok permutation might get invalidated later, but we
    // could still end up believing that unchecked ones are ok for this pair
    
  } // for i
  
  return any_matches;
  
} // TestPointPair

// note that for now, there is no subsuming
// this function will need to change if I want to add it
bool npt::SingleMatcher::TestHrectPair(const DHrectBound<2>& box1, 
                                       const DHrectBound<2>& box2,
                                       index_t tuple_ind_1, index_t tuple_ind_2,
                                       std::vector<bool>& permutation_ok) {
  
  bool any_matches = false;
  
  double max_dist_sq = box1.MaxDistanceSq(box2);
  double min_dist_sq = box1.MinDistanceSq(box2);
  
  // iterate over all the permutations
  for (index_t i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    index_t template_index_1 = GetPermIndex_(i, tuple_ind_1);
    index_t template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
    double upper_bound_sqr = upper_bounds_sqr_(template_index_1, 
                                               template_index_2);
    double lower_bound_sqr = lower_bounds_sqr_(template_index_1, 
                                               template_index_2);
    
    // are they too far or too close?
    if (max_dist_sq < lower_bound_sqr || min_dist_sq > upper_bound_sqr) {
     
      // this permutation doesn't work
      permutation_ok[i] = false;
      
    }
    else {
     
      // this permutation might work
      any_matches = true;
      
    } // end if
    
  } // for i
  
  return any_matches;
  
} // TestHrectPair()
