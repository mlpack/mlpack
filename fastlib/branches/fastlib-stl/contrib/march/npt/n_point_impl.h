/*
 *  n_point_impl.h
 *  
 *
 *  Created by William March on 4/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_IMPL_H
#define N_POINT_IMPL_H

// TODO: replace these with something better
#define EXCLUDE 0
#define INCONCLUSIVE 1
#define SUBSUME 2
#define BAD_SYMMETRY 3
#define MAX_FAC 10


#include "fastlib/fastlib.h"

// Combinatorial functions I'll need in more than one place
namespace n_point_impl {
 
  int NChooseR(int n, int r);
    
} // namespace


class Permutations {
  
private:
  
  GenMatrix<index_t> permutation_indices_;
  
  int num_perms_;
  
  int factorials_[MAX_FAC];
  
  // k is the column, permutation_index is the row
  void GeneratePermutations_(int k, int* permutation_index, 
                             GenVector<index_t>& trial_perm) {
    
    // the auton code doesn't do this
    if (*permutation_index >= num_perms_) {
      return;
    }
    
    for (index_t i = 0; i < trial_perm.length(); i++) {
      
      bool perm_ok = true;
      
      // has i been used in the permutation yet?
      for (index_t j = 0; perm_ok && j < k; j++) {
        
        if (trial_perm[j] == i) {
          perm_ok = false;
        }
        
      } // for j
      
      if (perm_ok) {
        
        trial_perm[k] = i;
        
        if (k == trial_perm.length() - 1) {
          
          permutation_indices_.CopyVectorToColumn(*permutation_index, 
                                                  trial_perm);
          (*permutation_index)++;
        } 
        else {
          
          GeneratePermutations_(k+1, permutation_index, trial_perm);
          
        }
        
      } // the permutation works
      
    } // for i
    
  } // GeneratePermutations
  
  
public:
  
  void Init(index_t n) {
    
    factorials_[0] = 1;
    for (index_t i = 1; i < MAX_FAC; i++) {
      factorials_[i] = i * factorials_[i-1];
    } // fill in factorials
    
    
    num_perms_ = factorials_[n];
    permutation_indices_.Init(n, num_perms_);
    
    // TODO: fill in the permutations
    GenVector<index_t> trial_perm;
    trial_perm.Init(n);
    for (index_t i = 0; i < n; i++) {
      trial_perm[i] = -1;
    }
    
    int index = 0;
    
    GeneratePermutations_(0, &index, trial_perm);
    
    /*
     #ifdef DEBUG
     printf("\n");
     for (index_t i = 0; i < n; i++) {
     
     for (index_t j = 0; j < num_perms_; j++) {
     
     printf("%d, ", permutation_indices_.ref(i, j));
     
     } // for j
     
     printf("\n");
     
     } // for i
     printf("\n");
     #endif
     */
    
    
  } // Init()
  
  int num_perms() {
    return num_perms_;
  }
  
  int factorials(int n) {
    DEBUG_ASSERT(n < MAX_FAC && n >= 0);
    return factorials_[n];
  }
  
  index_t GetPermutation(index_t perm_index, index_t pt_index) {
    
    // swapped to account for column major (not row major) format
    return permutation_indices_.get(pt_index, perm_index);
    
  } // GetPermutation
  
}; // class Permutations


#endif