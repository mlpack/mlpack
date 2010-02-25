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