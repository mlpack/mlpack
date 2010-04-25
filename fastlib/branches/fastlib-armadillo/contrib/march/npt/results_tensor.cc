/*
 *  n_point_results.cc
 *  
 *
 *  Created by William March on 4/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_results.h"

index_t ResultsTensor::FindIndex_(const ArrayList<index_t>& indices) {
  
  DEBUG_ASSERT(indices.length() == tuple_size_);
  
  // Do we assume that the indices are in order?  Let's do it for now, I think
  // they might be because the nodes will be ordered
  // What about the base case, though?
  
  ArrayList<index_t> sort_ind;
  sort_ind.InitCopy(indices);
  
  // TODO: double check that this is correct
  std::sort(sort_ind.begin(), sort_ind.end());
  
  // assuming the smallest index is first
  index_t return_ind = 0;
  for (index_t i = 0; i < tuple_size_; i++) {
    
    // TODO: this probably isn't correct
    return_ind += sort_ind[i] * pow(tuple_size_, i);
    
  } // for i
  
  return return_ind;
  
} // FindIndex_

void ResultsTensor::SetRange(const ArrayList<index_t>& lower_ind, 
                             const ArrayList<index_t>& upper_ind, int val) {
  
  
  
} // SetRange


void ResultsTensor::AddToRange(const ArrayList<index_t>& lower_ind, 
                               const ArrayList<index_t>& upper_ind, int val) {
  
  
  
  
} // AddToRange


// look at whole results tensor, fill in subsumes and excludes based on 
// the distances
void ResultsTensor::FillRows(index_t position_ind_1, index_t position_ind_2,
                             double max_dist_sq, double min_dist_sq) {
  
  // determine which ranges are excluded and which subsumed
  
  // fill in the entire sub-tensor where the pos_ind_1th 
  
} // FillRows
















