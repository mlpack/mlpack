/**
 *  multi_matcher.cc
 *  
 *
 *  Created by William March on 3/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "multi_matcher.h"

// fills in the ranges of indices in the bandwidth
void MultiMatcher::FindBandwidths_(double min_dist_sq, double max_dist_sq, 
                                   double* max_subsume, double* min_exclude) {

  
  
} // FindBandwidths_()


// this needs to return the correct range of inconclusive bandwidths for the
// pair of nodes (i.e. everything inside is subsumed, everything outside is 
// excluded)
// PROBLEM: the range for one permutation and that for another need not overlap
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


