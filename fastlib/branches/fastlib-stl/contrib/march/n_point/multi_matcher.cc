/*
 *  multi_matcher.cc
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "multi_matcher.h"

// Todo: think about whether it's worth keeping up with what we pruned before


index_t npt::MultiMatcher::IndexMatcherDim_(index_t i, index_t j) {
  
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
  
} // IndexMatcherDim

bool npt::MultiMatcher::TestNodeTuple(NodeTuple& nodes) {

  for (index_t i = 0; i < upper_bounds_sq_.size(); i++) {
    
    if (nodes.lower_bound(i) > upper_bounds_sq_[i]) {
      return false;
    }
    
    if (nodes.upper_bound(i) < lower_bounds_sq_[i]) {
      return false;
    }
    
  } // for i
  
  return true;
  
} // TestNodeTuple


bool npt::MultiMatcher::TestPointPair(double dist_sq, index_t new_ind, index_t old_ind,
                                 std::vector<bool>& permutation_ok,
                                 std::vector<std::vector<index_t> >&perm_locations) {

  bool any_matches = false;
  
  for (index_t perm_ind = 0; perm_ind < num_permutations_; perm_ind++) {
    
    if (!permutation_ok[perm_ind]) {
      continue;
    }
    
    index_t template_index_1 = GetPermIndex_(perm_ind, new_ind);
    index_t template_index_2 = GetPermIndex_(perm_ind, old_ind);
    
    std::vector<double>::iterator lo;
    std::vector<double>::iterator hi;
    
    double dist = std::sqrt(dist_sq);
    
    // template_index_1,2 determines which entry of matcher_dists we're dealing
    // with
    // Find the right entry of that matcher_dists using std::lower_bound
    // and std::upper_bound
    // Check that it works
    // If so, make perm_location[perm_ind]
    
    // which of the (n choose 2) dimensions of the results tensor are we 
    // dealing with in the current permutation
    index_t matcher_ind = IndexMatcherDim_(template_index_1, template_index_2);
    
    // TODO: double check where these are putting me
    lo = std::lower_bound(matcher_dists_[matcher_ind].begin(),
                          matcher_dists_[matcher_ind].end(), dist);
    //hi = std::upper_bound(matcher_dists_[matcher_ind].begin(),
    //                      matcher_dists_[matcher_ind].end(), dist_sq);
    
    double closest_matcher;
    index_t closest_ind;
    
    if (lo == matcher_dists_[matcher_ind].end()) {
      closest_matcher = matcher_dists_[matcher_ind].back();
      closest_ind = matcher_dists_[matcher_ind].size() - 1;
    }
    else if (lo == matcher_dists_[matcher_ind].begin()) {
      closest_matcher = matcher_dists_[matcher_ind].front();
      closest_ind = 0;
    }
    else {
      
      index_t low_ind = lo - matcher_dists_[matcher_ind].begin();
      
      double high_dist = matcher_dists_[matcher_ind][low_ind] - dist;
      double low_dist = dist  - matcher_dists_[matcher_ind][low_ind - 1];
      
      if (high_dist < low_dist) {
        closest_matcher = matcher_dists_[matcher_ind][low_ind];
        closest_ind = lo - matcher_dists_[matcher_ind].begin();
      }
      else {
        closest_matcher = matcher_dists_[matcher_ind][low_ind - 1];
        closest_ind = lo - 1 - matcher_dists_[matcher_ind].begin();        
      }
      
    }  // Figure out which matcher is closest
    
    // Now, check that the points actually fit that matcher
    // IMPORTANT: I'm assuming that if it doesn't fit the closest one, then
    // it won't fit any others, which seems reasonable
    // I think this does make some assumptions about no overlaps, though
    
    double high_dist = (closest_matcher + half_band_) 
                        * (closest_matcher + half_band_);
    double low_dist = (closest_matcher - half_band_) 
                        * (closest_matcher - half_band_);

    bool this_matches = (dist_sq <= high_dist) &&
                        (dist_sq >= low_dist);

    
    //std::cout << "Testing closest_ind: " << closest_ind << "\n";
    //std::cout << "Testing closest_matcher: " << closest_matcher << "\n";
    
    if (this_matches) {
      
      any_matches = true;
     // std::cout << "Setting perm locations to " << closest_ind << "\n";
      
      assert(closest_ind < matcher_dists_[matcher_ind].size());
      
      perm_locations[perm_ind][matcher_ind] = closest_ind;
      
    }
    else {
      permutation_ok[perm_ind] = false;
    }
    
  } // for perm_ind
  
  
  return any_matches;
  
} // TestPointPair



