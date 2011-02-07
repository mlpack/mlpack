/*
 *  permutations.h
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 *  Stores all the permutations of n elements.  Used in the standard 
 *  multi-tree algorithm.
 *
 */

#ifndef PERMUTATIONS_H
#define PERMUTATIONS_H

# include "fastlib/fastlib.h"

namespace npt {

  class Permutations {
    
  private:
    
    ///////////// member variables ///////////////////////
    
    
    // permutation_indices_(i, j) is the location of point i in permutation j
    arma::Mat<index_t> permutation_indices_;
    
    // the value of n being considered
    index_t tuple_size_;
    
    // tuple_size_!
    index_t num_perms_;
    
    // helper function at startup, actually forms all the permutations
    void GeneratePermutations_(int k, int* perm_index,
                               arma::Col<index_t>& trial_perm);
    
    
    
    
  public:
    
    // The constructor needs to fill in permuation_indices_
    Permutations(index_t n) {

      tuple_size_ = n;
      
      // compute the factorial
      num_perms_ = 1;
      for (index_t i = 2; i <= tuple_size_; i++) {
        num_perms_ = num_perms_ * i;
      } // for i
      
      arma::Col<index_t> trial_perm(tuple_size_);
      trial_perm.fill(-1);
      
      
      permutation_indices_.set_size(tuple_size_, num_perms_);
      permutation_indices_.fill(-1);
      //permutation_indices_(tuple_size_, 0);
      
      //std::cout << "n_rows: " << permutation_indices_.n_rows;
      //std::cout << " n_cols: " << permutation_indices_.n_cols << "\n";
      //std::cout << "trial_perm: " << trial_perm.n_rows << "\n";
      
      int perm_index = 0;
      //Print();
      GeneratePermutations_(0, &perm_index, trial_perm);
      
    } // constructor
    
    // just accesses elements of permutation_indices_
    index_t GetPermutation(index_t perm_index, index_t point_index) const {
      
      // note that these are backward from how they're input
      // the old code does it this way and I don't want to get confused
      return permutation_indices_(point_index, perm_index);
      
    } // GetPermutation()
    
    
    // for debugging purposes
    void Print() {
      
      permutation_indices_.print("Permutation Indices:");
      
    }
    
    
  }; // class
  
} // namespace

#endif

