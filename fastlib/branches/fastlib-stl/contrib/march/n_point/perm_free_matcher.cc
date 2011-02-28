/*
 *  perm_free_matcher.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "perm_free_matcher.h"



bool npt::PermFreeMatcher::TestNodeTuple(NodeTuple& nodes) {
  
  for (index_t i = 0; i < upper_bounds_sq_.size(); i++) {
    
    if (nodes.lower_bound(i) > upper_bounds_sq_[i]) {
      return false;
    }

    if (nodes.upper_bound(i) < lower_bounds_sq_[i]) {
      return false;
    }
    
  } // for i
  
  return true;
  
} // TestNodeTuple




bool npt::PermFreeMatcher::TestPointPair(double dist_sq, index_t tuple_index_1, 
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
    
    
    if (dist_sq <= upper_bounds_sq_mat_.ref(template_index_1, template_index_2)) {
      any_matches = true;
    } // this placement works
    else {
      permutation_ok[i] = false;
    } // this placement doesn't work
    
  } // for i
  
  return any_matches;  
  
  
} // TestPointPair


