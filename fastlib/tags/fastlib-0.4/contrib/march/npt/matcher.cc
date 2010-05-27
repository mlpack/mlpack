/*
 *  matcher.cc
 *  
 *
 *  Created by William March on 2/23/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "matcher.h"


bool Matcher::CheckDistances_(double dist_sq, index_t ind1, index_t ind2) {
  
  double upper_bd = upper_bounds_sqr_.ref(ind1, ind2);
  double lower_bd = lower_bounds_sqr_.ref(ind1, ind2);
  
  return (dist_sq <= upper_bd && dist_sq >= lower_bd);
  
} // CheckDistances_()


// TODO: should I check for subsuming prunes or not?
// the auton code doesn't seem to
int Matcher::TestHrectPair(const DHrectBound<2>& box1, const DHrectBound<2>& box2,
                           index_t tuple_index_1, index_t tuple_index_2,
                           ArrayList<int>& permutation_ok) {
  
  // we'll only return EXCLUDE if all the permutations are excluded
  int result = EXCLUDE;

  double min_dist_sq = box1.MinDistanceSq(box2);
  double max_dist_sq = box1.MaxDistanceSq(box2);
  
  for (index_t i = 0; i < num_permutations_; i++) {
    
    if (permutation_ok[i] == EXCLUDE) {
      continue;
    } // does this permutation work?
    
    index_t template_index_1 = GetPermutationIndex_(i, tuple_index_1);
    index_t template_index_2 = GetPermutationIndex_(i, tuple_index_2);
    
    
    double upper_bound_sq = upper_bounds_sqr_.ref(template_index_1, 
                                                  template_index_2);
    double lower_bound_sq = lower_bounds_sqr_.ref(template_index_1, 
                                                  template_index_2);
    
    if (max_dist_sq < lower_bound_sq || min_dist_sq > upper_bound_sq) {
      // this permutation is excluded, but others might work
      
      permutation_ok[i] = EXCLUDE;
      
    } // the points are too close together or too far apart
    else if (min_dist_sq >= lower_bound_sq && max_dist_sq <= upper_bound_sq) {
      // subsume
      
      result = SUBSUME;
      
    } // the bound is subsumed
    else {
      // inconclusive
      
      result = INCONCLUSIVE;
      
    } // can't tell
    
  } // for i
  
  return result;
  
} // TestHrectPair()




// IMPORTANT: this assumes that the two points have already been checked for 
// symmetry
bool Matcher::TestPointPair(double dist_sq, index_t tuple_index_1, 
                            index_t tuple_index_2, 
                            ArrayList<bool>& permutation_ok) {

  DEBUG_ASSERT(permutation_ok.size() == perms_.factorials(tuple_size_));

  bool any_matches = false;
  
  for (index_t i = 0; i < num_permutations_; i++) {
    
    if (!(permutation_ok[i])) {
      continue;
    } // does this permutation work?
    
    index_t template_index_1 = GetPermutationIndex_(i, tuple_index_1);
    index_t template_index_2 = GetPermutationIndex_(i, tuple_index_2);
    
   if (CheckDistances_(dist_sq, template_index_1, 
                       template_index_2)) {
     any_matches = true;
   } // this placement works
   else {
     permutation_ok[i] = false;
   } // this placement doesn't work
    
  } // for i
  
  return any_matches;
  
} // TestPointPair()