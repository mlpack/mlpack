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

  std::vector<double> node_lower_;
  std::vector<double> node_upper_;
  
  for (int i = 0; i < nodes.tuple_size(); i++) {
    
    for (int j = i+1; j < nodes.tuple_size(); j++) {
      
      double lo = nodes.node_list(i).bound().MinDistanceSq(nodes.node_list(j));
      double hi = nodes.node_list(i).bound().MaxDistanceSq(nodes.node_list(j));
      
      node_lower_.push_back(lo);
      node_upper_.push_back(hi);
      
    }
    
  }

  std::sort(node_lower_.begin(), node_lower_.end());
  std::sort(node_upper_.begin(), node_upper_.end());

  for (index_t i = 0; i < upper_bounds_sq_.size(); i++) {
    
    if (node_lower_[i] > upper_bounds_sq_[i]) {
      return false;
    }
    
    if (node_upper_[i] < lower_bounds_sq_[i]) {
      return false;
    }
    
  } // for i
  
  return true;
  
} // TestNodeTuple


bool npt::MultiMatcher::TestPointPair_(double dist_sq, index_t new_ind, index_t old_ind,
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
    //std::vector<double>::iterator hi;
    
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


void npt::MultiBandwidthAlg::BaseCaseHelper_(
                                             std::vector<std::vector<index_t> >& point_sets,
                                             std::vector<bool>& permutation_ok,
                                             std::vector<std::vector<index_t> >& perm_locations,
                                             std::vector<index_t>& points_in_tuple,
                                             int k) {
  
  
  // perm_locations[i][j] = k means that in the ith permutation, that 
  // matcher_dists_[j][k] is the current entry in the matcher that this tuple
  // satisfies
  
  std::vector<bool> perm_ok_copy(permutation_ok);
  std::vector<std::vector<index_t> > perm_locations_copy(perm_locations);
  
  bool bad_symmetry = false;
  
  // iterate over possible new points
  for (index_t i = 0; i < point_sets[k].size(); i++) {
    
    index_t new_point_ind = point_sets[k][i];
    bool this_point_works = true;
    
    bad_symmetry = false;
    
    bool i_is_random = (k < num_random_);
    
    arma::colvec new_point_vec;
    
    if (i_is_random) {
      new_point_vec = random_mat_.col(new_point_ind);
    } 
    else {
      new_point_vec = data_mat_.col(new_point_ind);
    }
    
    // copy the permutation 
    perm_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
    
    // TODO: check if I can accurately copy this more directly
    for (index_t m = 0; m < perm_locations_copy.size(); m++) {
      perm_locations_copy[m].assign(perm_locations[m].begin(), 
                                    perm_locations[m].end());
    } // for m
    
    // TODO: double check that I can exit on bad symmetry here
    for (index_t j = 0; j < k && this_point_works && !bad_symmetry; j++) {
      
      index_t old_point_ind = points_in_tuple[j];
      
      bool j_is_random = (j < num_random_);
      
      bad_symmetry = (i_is_random == j_is_random) 
      && (new_point_ind <= old_point_ind);
      
      // TODO: if bad_symmetry, can I break out of the loop?
      if (!bad_symmetry) {
        
        arma::colvec old_point_vec;
        if (j_is_random) {
          old_point_vec = random_mat_.col(old_point_ind);
        }
        else {
          old_point_vec = data_mat_.col(old_point_ind);
        }
        
        double point_dist_sq = la::DistanceSqEuclidean(old_point_vec, 
                                                       new_point_vec);
        
        this_point_works = TestPointPair_(point_dist_sq, j, k, 
                                          perm_ok_copy,
                                          perm_locations_copy);
        // perm_locations_copy should now be filled in 
        
      } // check symmetry
      
    } // check existing points
    
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = new_point_ind;
      
      if (k == tuple_size_ - 1) {
        
        // fill in all the results that worked
        
        std::set<index_t> results_set;
        
        for (index_t n = 0; n < perm_locations_copy.size(); n++) {
          
          if (perm_ok_copy[n]) {
            index_t results_ind = FindResultsInd_(perm_locations_copy[n]);
            results_set.insert(results_ind);
            //std::cout << "Inserting: " << results_ind << "\n";
          }
        } // for n
        
        // Now, iterate through all (distinct) results keys in the set and add
        // them to the total
        std::set<index_t>::iterator it;
        
        for (it = results_set.begin(); it != results_set.end(); it++) {
          
          results_[num_random_][*it]++;
          
        }
        
        double this_weight = 1.0;
        for (int tuple_ind = 0; tuple_ind < num_random_; tuple_ind++) {
          this_weight *= random_weights_(points_in_tuple[tuple_ind]);
        }
        for (index_t tuple_ind = num_random_; tuple_ind < tuple_size_; 
             tuple_ind++) {
          
          this_weight *= data_weights_(points_in_tuple[tuple_ind]);
          
        } // iterate over the tuple
        
        for (it = results_set.begin(); it != results_set.end(); it++) {
          weighted_results_[num_random_][*it] += this_weight;
        } 
        
      }
      else {
        
        BaseCaseHelper_(point_sets, perm_ok_copy, perm_locations_copy,
                        points_in_tuple, k+1);
        
      }
      
    } // do we still need to work with these points?
    
  } // iterate over possible new points
  
  
} // BaseCaseHelper_



void npt::MultiMatcher::BaseCase(NodeTuple& nodes) {
  
  std::vector<std::vector<index_t> > point_sets(tuple_size_);
  
  for (index_t node_ind = 0; node_ind < tuple_size_; node_ind++) {
    
    point_sets[node_ind].resize(nodes.node_list(node_ind)->count());
    
    for (index_t i = 0; i < nodes.node_list(node_ind)->count(); i++) {
      
      point_sets[node_ind][i] = i + nodes.node_list(node_ind)->begin();
      
    } // for i
    
  } // for node_ind
  
  std::vector<bool> permutation_ok(num_permutations_, true);
  
  std::vector<index_t> points_in_tuple(tuple_size_, -1);
  
  std::vector<std::vector<index_t> > perm_locations(num_permutations_);
  
  for (index_t i = 0; i < perm_locations.size(); i++) {
    perm_locations[i].resize(num_bands_.size(), INT_MAX);
  }
  
  BaseCaseHelper_(point_sets, permutation_ok, perm_locations, 
                  points_in_tuple, 0);
  
  
  
} // BaseCase






