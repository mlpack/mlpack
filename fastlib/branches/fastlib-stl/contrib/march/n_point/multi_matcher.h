/*
 *  multi_matcher.h
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


// Idea: for each of the (n choose 2) distances in the matcher, the user will 
// specify a range and number of distances to compute for 


// IMPORTANT: assuming that all dimensions have the same thickness
// assuming that matcher values +- band don't overlap within a dimension

#ifndef MULTI_MATCHER_H
#define MULTI_MATCHER_H

#include "permutations.h"
#include "node_tuple.h"

namespace npt {
  
  class MultiMatcher {
    
  private:

    // for now, I'm assuming a single, global thickness for each dimension of 
    // the matcher
    double bandwidth_;
    double half_band_;
    
    int tuple_size_;
    
    int n_choose_2_;
    
    // IMPORTANT: need an ordering of these
    
    // all these need length (n choose 2)
    // these are the max and min of the range for each dimension
    std::vector<double> min_bands_sq_;
    std::vector<double> max_bands_sq_;
    
    // entry i,j is the jth matcher value in dimension i
    std::vector<std::vector<double> > matcher_dists_; 
    
    /*
    // (i,j) is true if matcher dimensions i and j are equal
    std::vector<std::vector<bool> > matcher_dim_equal_;
    */
    
    // the number of bandwidths in each dimension
    std::vector<int> num_bands_;
    
    // these are just min_bands and max bands sorted (and squared)
    std::vector<double> upper_bounds_sq_;
    std::vector<double> lower_bounds_sq_;
    
    Permutations perms_;
    int num_permutations_;
    
    
    // We want the matcher dimension that is the distance between point i and j
    index_t IndexMatcherDim_(index_t i, index_t j);
    
    index_t GetPermIndex_(index_t perm_index, index_t pt_index) {
      return perms_.GetPermutation(perm_index, pt_index);
    } // GetPermIndex_
    
    
  public:
    
    MultiMatcher(const std::vector<double>& min_bands, 
                 const std::vector<double>& max_bands,
                 const std::vector<int>& num_bands, 
                 const double band, index_t tuple_size) : num_bands_(num_bands),
    perms_(tuple_size), tuple_size_(tuple_size)
    {

      bandwidth_ = band;
      half_band_ = bandwidth_ / 2.0;
      
      min_bands_sq_.resize(min_bands.size());
      max_bands_sq_.resize(max_bands.size());

      upper_bounds_sq_.resize(min_bands.size());
      lower_bounds_sq_.resize(max_bands.size());
      
      for (index_t i = 0; i < max_bands.size(); i++) {
        
        min_bands_sq_[i] = min_bands[i] * min_bands[i];
        max_bands_sq_[i] = max_bands[i] * max_bands[i];
        
        if (min_bands[i] - half_band_ > 0) {
          lower_bounds_sq_[i] = (min_bands[i] - half_band_) 
                                * (min_bands[i] - half_band_);
        
        }
        else {
          lower_bounds_sq_[i] = 0.0;
        }
          upper_bounds_sq_[i] = (max_bands[i] + half_band_) 
                              * (max_bands[i] + half_band_);
        
      }
      
      std::sort(lower_bounds_sq_.begin(), lower_bounds_sq_.end());
      std::sort(upper_bounds_sq_.begin(), upper_bounds_sq_.end());
      
      n_choose_2_ = num_bands_.size();
      matcher_dists_.resize(n_choose_2_);
      
      for (index_t i = 0; i < n_choose_2_; i++) {
        
        double band_step = (max_bands[i] - min_bands[i]) / ((double)num_bands_[i] - 1.0);
        
        matcher_dists_[i].resize(num_bands_[i]);
        
        if (num_bands_[i] > 1) {
          for (index_t j = 0; j < num_bands_[i]; j++) {
            
            matcher_dists_[i][j] = min_bands[i] + (double)j * band_step;
            
          } // for j
        }
        else {
          matcher_dists_[i][0] = min_bands[i];
        }
      } // for i

      
      num_permutations_ = perms_.num_permutations();
      
    } // constructor
    
    
    bool TestPointPair(double dist_sq, index_t new_ind, index_t old_ind,
                       std::vector<bool>& permutation_ok,
                       std::vector<std::vector<index_t> >&perm_locations);
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    
    int num_permutations() {
      return perms_.num_permutations(); 
    }
    
    double matcher_dists(index_t i, index_t j) {
      return (matcher_dists_[i][j]);
    }
    
  }; // class
  
} // namespace

#endif 
