/**
 *  multi_matcher.cc
 *  
 *
 *  Created by William March on 3/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "multi_matcher.h"


bool MultiMatcher::TestPointPair(double dist_sq, index_t tuple_index_1, 
                                 index_t tuple_index_2, 
                                 ArrayList<bool>& permutation_ok,
                                 ArrayList<GenMatrix<index_t> >& permutation_ranges) {
  
  bool this_point_works = false;
  
  DEBUG_ASSERT(tuple_index_1 < tuple_index_2);
  
  for (index_t perm_ind = 0; perm_ind < num_permutations_; perm_ind++) {
    
    // this permutation is already bad
    if (! permutation_ok[perm_ind]) {
      continue;
    }
    
    if (dist_sq >= distances_[num_bins_ - 1]) {
        
      // this permutation is bad
      permutation_ok[perm_ind] = false;
      continue;
      
    }
    
    // figure out what the largest index that works is here
    //permutation_ranges[perm_ind].set(tuple_index_1, tuple_index_2, dist_sq);
    // TODO: do I need to set the other side of the diagonal?
    
    // TODO: double check this
    double* ind_ptr = std::upper_bound(distances_.begin(), distances_.end(), 
                                       dist_sq);
    int ind = (int)(ind_ptr - distances_.begin());
    permutation_ranges[perm_ind].set(tuple_index_1, tuple_index_2, ind);
    
    this_point_works = true;
    
  } // loop over permutations
  
  return this_point_works;
  
  
} // TestPointPair



/*
void MultiMatcher::TestNodes_(const DHrectBound<2>& box1, 
                              const DHrectBound<2>& box2,
                              index_t tuple_index_1, index_t tuple_index_2,
                              perm_ind, DRange range_out) {
  
  double max_dist_sq = box1.MaxDistanceSq(box2);
  double min_dist_sq = box1.MinDistanceSq(box2);

  double max_subsume, min_exclude;
  FindBandwidths_(min_dist_sq, max_dist_sq, &max_subsume, &min_exclude);
  
  // these are the indices in the results matrix
  index_t template_index_1 = GetPermutationIndex_(perm_ind, tuple_index_1);
  index_t template_index_2 = GetPermutationIndex_(perm_ind, tuple_index_2);
  
  range_out.Init(max_subsume, min_exclude);
    
} // TestNodes_
*/

