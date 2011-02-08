/*
 *  permutations.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "permutations.h"

// TODO: try a reference?
void npt::Permutations::GeneratePermutations_(int k, int* perm_index,
                                              arma::Col<index_t>& trial_perm) {

  // simple bounds check
  if (*perm_index >= num_perms_) {
    return;
  } 

  // Iterate over all points (i.e. everything that might be in the permutation)
  for (index_t i = 0; i < tuple_size_; i++) {
    
    bool perm_ok = true;
    
    // Iterate over everything already in trial_perm
    for (index_t j = 0; perm_ok && j < k; j++) {
      
      // Did we already use j in this one?  Then don't use it again.
      if (trial_perm[j] == i) {
        perm_ok = false;
      }
      
    } // for j
    
    // go to the next i if this one didn't work, otherwise proceed
    if (perm_ok) {
      
      // add i to the trial permutation
      trial_perm(k) = i;
    
      // if the whole permutation is filled, put it in the matrix
      if (k == tuple_size_ - 1) {
        
        permutation_indices_.col(*perm_index) = trial_perm;
        (*perm_index)++;
        
      } // is the permutation filled?
      else {
        // move on to the next spot in the permutation
        GeneratePermutations_(k+1, perm_index, trial_perm);
        
      } // permutation not filled
        
    } // if perm_ok
    
  } // for i
  
  
} // GeneratePermutations_

