/*
 *  n_point_results.cc
 *  
 *
 *  Created by William March on 4/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "results_tensor.h"

// This is the strictly upper triangular version
/*
index_t ResultsTensor::FindIndex_(const ArrayList<index_t>& indices) {
  

  // This should be unnecessary

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

*/

// Full tensor version
index_t ResultsTensor::FindIndex_(const ArrayList<index_t>& indices) {
  
  index_t result = indices[0];
  index_t power = num_bins_;
  
  for (index_t i = 1; i < indices.size(); i++) { 
    
    result += indices[i] * power;
    power = power * num_bins_;
    
  } // for i
  
  return result;
  
} // FindIndex_()

bool ResultsTensor::IncrementIndex_(ArrayList<index_t>& new_ind, 
                                    const ArrayList<index_t>& orig_ind, 
                                    index_t k) {
  
  if (k >= tensor_rank_) {
    return true;
  }
  
  new_ind[k]++;
  if (new_ind[k] >= num_bins_) {
    new_ind[k] = orig_ind[k];
    return IncrementIndex_(new_ind, orig_ind, k+1);
  }
  else {
    return false;
  }
  
} // IncrementIndex


void ResultsTensor::IncrementRange(const GenMatrix<index_t>& lower_inds) {
  
  // map row major into the index array
  // TODO: would column major be more efficient?
  
  
  
  ArrayList<index_t> this_result;
  this_result.Init(tensor_rank_);
  
  index_t row_ind = 0;
  index_t col_ind = 1;
  
  for (index_t i = 0; i < this_result.size(); i++) {
    
    this_result[i] = lower_inds.get(row_ind, col_ind);
    DEBUG_ASSERT(this_result[i] < num_bins_ && this_result[i] >= 0);
    col_ind++;
    
    if (col_ind >= tuple_size_) {
      row_ind++;
      col_ind = row_ind+1;
    } // are we at the end of the row?
    
  } // for i
  
  ArrayList<index_t> this_result_orig;
  this_result_orig.InitCopy(this_result);
  
  bool done = false;
  
  while (!done) {
    
    // fill in the entry
    
    index_t this_ind = FindIndex_(this_result);
    
    if (!filled_results_[this_ind]) {
      filled_results_[this_ind] = true;
      results_[this_ind]++;
    }
    
    // increment the array
    done = IncrementIndex_(this_result, this_result_orig, 0);
    
  } // while
  
} // IncrementRange()















