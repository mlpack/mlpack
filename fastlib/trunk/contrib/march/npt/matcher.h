/*
 *  matcher.h
 *  
 *
 *  Created by William March on 2/23/10.
 *
 *
 *  Stores the upper and lower bounds on each pair in the n-tuple.
 *  Can also evaluate whether a set of points or nodes violates these 
 *  conditions.
 */

// TODO: where should I keep the list of acceptable permutations?  
// I think it should just be a vector passed down in the recursion

#ifndef MATCHER_H
#define MATCHER_H

#include "fastlib/fastlib.h"

#define MAX_FAC 10

class Matcher {
  
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
  
  // these are symmetric n \times n matrices
  // the diagonals are not defined, since they are never accessed.
  // for a tuple to work, we need L_{i,j} \leq d(x_i, x_j) \leq H_{i,j}
  // for all pairs (x_i, x_j) in the tuple (under some permutation)
  // these are the squared bounds to prevent square roots in the code
  Matrix lower_bounds_sqr_;
  Matrix upper_bounds_sqr_;
  
  Permutations perms_;
  
  int tuple_size_;
  int num_permutations_;
  
  
  ///////////////////// functions ///////////////////////
  
  /**
   *
   */
  index_t GetPermutationIndex_(index_t perm_index, index_t pt_index) {
    
    return perms_.GetPermutation(perm_index, pt_index);
    
  } // GetPermutation
  
  /**
   *
   */
  bool CheckDistances_(double dist_sq, index_t ind1, index_t ind2);

  
public:
  
  /**
   *
   */
  void Init(const Matrix& lower, const Matrix& upper, int tuple_size) {
    
    lower_bounds_sqr_.Copy(lower);
    upper_bounds_sqr_.Copy(upper);
    tuple_size_ = tuple_size;
    
    // need to set them to be the squares of the values read in 
    for (index_t i = 0; i < tuple_size_; i++) {
      for (index_t j = 0; j < i; j++) {
        
        double new_low = lower_bounds_sqr_.get(i, j) * lower_bounds_sqr_.get(i, j);
        double new_hi = upper_bounds_sqr_.get(i, j) * upper_bounds_sqr_.get(i, j);
        
        lower_bounds_sqr_.set(i, j, new_low);
        lower_bounds_sqr_.set(j, i, new_low);
        
        upper_bounds_sqr_.set(i, j, new_hi);
        upper_bounds_sqr_.set(j, i, new_hi);
        
      } // for j
    } // for i
    
    perms_.Init(tuple_size_);
    num_permutations_ = perms_.num_perms();
    
    
  } // Init()
  
  int num_permutations() const {
    return num_permutations_;
  }
  
  
  /**
   *
   */
  bool TestPointPair(double dist_sq, index_t tuple_index_1, 
                     index_t tuple_index_2, ArrayList<bool>& permutation_ok);
  
  
  // TODO: how will I know what the nodes are inside this function?
  // I think the Auton code just passes the hrects
  /**
   *
   */
  bool TestNodePair();
  
}; // Matcher

#endif

