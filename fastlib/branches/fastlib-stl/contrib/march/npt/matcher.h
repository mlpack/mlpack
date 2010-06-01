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
#include "n_point_impl.h"

class Matcher {
  
private:
  
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
  
  
  /**
   *
   */
  int TestHrectPair(const DHrectBound<2>& box1, const DHrectBound<2>& box2,
                    index_t tuple_index_1, index_t tuple_index_2,
                    ArrayList<int>& permutation_ok);
    
}; // Matcher

#endif

