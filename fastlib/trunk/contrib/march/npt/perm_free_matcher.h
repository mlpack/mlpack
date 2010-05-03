/*
 *  perm_free_matcher.h
 *  
 *
 *  Created by William March on 4/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef PERM_FREE_MATCHER_H
#define PERM_FREE_MATCHER_H

#ifndef EXCLUDE
#define EXCLUDE 0
#define INCONCLUSIVE 1
#define SUBSUME 2
#define BAD_SYMMETRY 3
#define MAX_FAC 10
#endif

#include "n_point_nodes.h"

// TODO: still need permutations for the base case, I think

class PermFreeMatcher {
  
private:
  
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
  
  
  Matrix upper_bounds_sq_mat_;
  
  ArrayList<double> upper_bounds_sq_;
  
  index_t tuple_size_;
  
  Permutations perms_;
  int num_permutations_;
  
  
public:
  
  index_t GetPermutationIndex_(index_t perm_index, index_t pt_index) {
    
    return perms_.GetPermutation(perm_index, pt_index);
    
  } // GetPermutation
  
  int num_permutations() {
    return num_permutations_;
  }
  
  // assuming the input isn't squared or sorted
  void Init(const Matrix& mat_in, ArrayList<double>& upper_in,
            index_t size_in) {
    
    upper_bounds_sq_mat_.Init(mat_in.n_rows(), mat_in.n_cols());
    for (index_t i = 0; i < mat_in.n_rows(); i++) {
      for (index_t j = 0; j < mat_in.n_cols(); j++) {
        upper_bounds_sq_mat_.set(i, j, mat_in.get(i, j) * mat_in.get(i, j));
      }
    }
    
    tuple_size_ = size_in;
    
    upper_bounds_sq_.Init(upper_in.size());
    
    for (index_t i = 0; i < upper_in.size(); i++) {
      
      upper_bounds_sq_[i] = upper_in[i] * upper_in[i];
      
    } // for i
    
    std::sort(upper_bounds_sq_.begin(), upper_bounds_sq_.end());
  
    perms_.Init(tuple_size_);
    num_permutations_ = perms_.num_perms();
    
    
  } // Init()
  
  bool TestPointPair(double dist_sq, index_t tuple_index_1, 
                     index_t tuple_index_2, 
                     ArrayList<bool>& permutation_ok);
  
  
  int CheckNodes(NodeTuple& nodes);
  
  
}; // PermFreeMatcher



#endif


