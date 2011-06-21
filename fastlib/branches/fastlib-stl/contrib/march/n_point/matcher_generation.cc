/*
 *  matcher_generation.cc
 *  
 *
 *  Created by William March on 6/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "matcher_generation.h"

index_t npt::MatcherGenerator::FindWhichMatcher_(index_t i, index_t j) {
  
  if (i > j) {
    std::swap(i, j);
  }
  
  assert(i != j);
  
  index_t res = 0;
  
  if (i > 0) {
    for (index_t k = 0; k < i; k++) {
      res += (tuple_size_ - k - 1);
    }
  }
  
  res += (j - i - 1);
  
  return res;
  
} 

void npt::MatcherGenerator::FillInMatchers_(std::vector<index_t>& matcher_ind, 
                                            int k) {

  // we're filling in the kth spot in matcher ind
  
  std::vector<index_t>& matcher_ind_copy(matcher_ind);
  
  for (index_t i = 0; i < num_bands_[k]; i++) {
    
    // Do I need to copy it again here?
    
    matcher_ind_copy[k] = i;
  
    if (k == matcher_ind.size() - 1) {
      // we've completed a tuple, make the matcher and add to the list
      
      arma::mat matcher(tuple_size_, tuple_size_);
      
      for (index_t m = 0; m < tuple_size_; m++) {
        
        matcher(m,m) = 0.0;
        
        for (index_t n = m+1; n < tuple_size_; n++) {
          
          index_t which_matcher = FindWhichMatcher_(m, n);
          
          matcher(m,n) = matcher_dists_[which_matcher][matcher_ind[which_matcher]];
          matcher(n,m) = matcher(m,n);
          
        } // for n
        
      } // for m
      
      matchers_.push_back(matcher);
      
    } // finished this matcher
    else {
      
      FillInMatchers_(matcher_ind_copy, k+1);
      
    } // keep recursing
    
  } // for i
  
} // FillInMatchers_
