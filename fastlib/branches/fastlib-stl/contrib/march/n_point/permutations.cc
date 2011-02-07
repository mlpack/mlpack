/*
 *  permutations.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "permutations.h"

void npt::Permutations::GeneratePermutations_(int k, int* perm_index,
                                              arma::Col<index_t>& trial_perm) {

  // simple bounds check
  if (*perm_index >= num_perms_) {
    return;
  } 

  // fill in all of trial perm
  for (index_t i = 0; i < tuple_size_; i++) {
    
    bool perm_ok = true;
    
    // check to see if i has already been used in this permutation
    for (index_t j = 0; perm_ok && j < k; j++) {
      
      if (trial_perm[j] == i) {
        perm_ok = false;
      }
      
    } // for j
    
    // go to the next i if this one didn't work, otherwise proceed
    if (perm_ok) {
      trial_perm(k) = i;
    
    
      // if the whole permutation is filled, put it in the matrix
      if (k == tuple_size_ - 1) {
        
        // not sure if this works yet, needs to be tested
        //permutation_indices_.insert_cols(permutation_indices_.n_cols, 
        //                                 trial_perm);
        permutation_indices_.col(*perm_index) = trial_perm;
        (*perm_index)++;
        
      } // is the permutation filled?
      else {
        
        GeneratePermutations_(k+1, perm_index, trial_perm);
        
      } 
        
    } // if perm_ok
    
  } // for i
  
  
} // GeneratePermutations_