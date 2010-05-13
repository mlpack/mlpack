/*
 *  n_point_results.cc
 *  
 *
 *  Created by William March on 4/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_results.h"

// TODO: test me!
index_t ResultsTensor::FindIndex_(const ArrayList<index_t>& indices) {
  
  ArrayList<index_t> sort_ind;
  sort_ind.InitCopy(indices);
  
  // TODO: double check that this is correct
  std::sort(sort_ind.begin(), sort_ind.end());
  
  // assuming the smallest index is first
  index_t return_ind = 0;
  for (index_t i = 1; i <= tensor_rank_; i++) {
    
    index_t this_prod = 1;
    
    for (index_t j = 0; j < i; j++) {
      
      this_prod *= indices[i-1] + j;
      
    } // for j
    
    this_prod = this_prod / n_point_impl::factorial(i);
    
    return_ind += this_prod;
    
  } // for i
  
  return return_ind;
  
} // FindIndex_


// There is a range for each of the n! permutations
void ResultsTensor::FillRanges(ArrayList<DRange>& ranges) {
  
  
  
} // FillRanges
















