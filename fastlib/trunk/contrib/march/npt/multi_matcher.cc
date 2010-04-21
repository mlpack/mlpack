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
                                   DRange* subsumed, DRange* excluded, 
                                   DRange* inconclusive) {

  
  
} // FindBandwidths_()



void MultiMatcher::TestNodes_(const DHrectBound<2>& box1, 
                              const DHrectBound<2>& box2,
                              index_t tuple_index_1, index_t tuple_index_2,
                              ArrayList<ResultsTensor>& permutation_ok) {
  
  double max_dist_sq = box1.MaxDistanceSq(box2);
  double min_dist_sq = box1.MinDistanceSq(box2);

  // where does it fit?  
  
  DRange subsumed, excluded, inconclusive;
  FindBandwidths_(min_dist_sq, max_dist_sq, &subsumed, &excluded, 
                  &inconclusive);
  
  for (index_t i = 0; i < num_permutations; i++) {
    
    ResultsTensor& results_i = permutation_ok[i];
    
    // these are the indices in the results matrix
    index_t template_index_1 = GetPermutationIndex_(i, tuple_index_1);
    index_t template_index_2 = GetPermutationIndex_(i, tuple_index_2);
    
    // this needs 
    results_i.FillRowsStatus(template_index_1, template_index_2, status);
    
  } // iterate over permutations
  
  
  
  
} // TestNodes_






