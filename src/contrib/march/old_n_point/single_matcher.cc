/*
 *  single_matcher.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "single_matcher.h"

bool npt::SingleMatcher::CheckDistances_(double dist_sq, size_t ind1, 
                                         size_t ind2) {
  
  return (dist_sq <= upper_bounds_sqr_(ind1, ind2) && 
          dist_sq >= lower_bounds_sqr_(ind1, ind2));
  
} //CheckDistances_


// note that this assumes that the points have been checked for symmetry
bool npt::SingleMatcher::TestPointPair_(double dist_sq, size_t tuple_ind_1, 
                                        size_t tuple_ind_2,
                                        std::vector<bool>& permutation_ok) {

  bool any_matches = false;
  
  // iterate over all the permutations
  for (size_t i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    size_t template_index_1 = GetPermIndex_(i, tuple_ind_1);
    size_t template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
    //std::cout << "template_indices_checked\n";
    
    // Do the distances work?
    if (CheckDistances_(dist_sq, template_index_1, template_index_2)) {
      any_matches = true;
    }
    else {
      permutation_ok[i] = false;
    }
    
    // IMPORTANT: we can't exit here if any_matches is true
    // This is because the ok permutation might get invalidated later, but we
    // could still end up believing that unchecked ones are ok for this pair
    
  } // for i
  
  return any_matches;
  
} // TestPointPair

// note that for now, there is no subsuming
// this function will need to change if I want to add it
// This returns true if the pair might satisfy the matcher
bool npt::SingleMatcher::TestHrectPair_(const DHrectBound<2>& box1, 
                                        const DHrectBound<2>& box2,
                                        size_t tuple_ind_1, size_t tuple_ind_2,
                                        std::vector<bool>& permutation_ok) {
  
  bool any_matches = false;
  
  double max_dist_sq = box1.MaxDistanceSq(box2);
  double min_dist_sq = box1.MinDistanceSq(box2);
  
  // iterate over all the permutations
  for (size_t i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    size_t template_index_1 = GetPermIndex_(i, tuple_ind_1);
    size_t template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
    double upper_bound_sqr = upper_bounds_sqr_(template_index_1, 
                                               template_index_2);
    double lower_bound_sqr = lower_bounds_sqr_(template_index_1, 
                                               template_index_2);
    
    /*
    printf("max_dist_sq: %g\n", max_dist_sq);
    printf("min_dist_sq: %g\n", min_dist_sq);
    printf("upper_bound_sq: %g\n", upper_bound_sqr);
    printf("lower_bound_sq: %g\n", lower_bound_sqr);
    */
    
    // are they too far or too close?
    if (max_dist_sq < lower_bound_sqr || min_dist_sq > upper_bound_sqr) {
     
      // this permutation doesn't work
      permutation_ok[i] = false;
      
    }
    else {
     
      // this permutation might work
      any_matches = true;
      
    } // end if
    
  } // for i
  
  return any_matches;
  
} // TestHrectPair()



bool npt::SingleMatcher::TestNodeTuple(NodeTuple& nodes) {
  
  bool possibly_valid = true;
  
  // note that this has to enforce symmetry
  std::vector<bool> permutation_ok(num_permutations_, true);
  
  // iterate over all nodes
  // IMPORTANT: right now, I'm exiting when I can prune
  // I need to double check that this works
  for (size_t i = 0; possibly_valid && i < tuple_size_; i++) {
    
    bool i_is_random = i < num_random_;
    
    NptNode* node_i = nodes.node_list(i);
    
    // iterate over all nodes > i
    for (size_t j = i+1; possibly_valid && j < tuple_size_; j++) {
      
      bool j_is_random = j < num_random_;
      
      NptNode* node_j = nodes.node_list(j);
      
      // check for symmetry
      // don't need this anymore, since the NodeTuple/recursive call will do it
 
      /*
      if (node_j->end() <= node_i->begin() && (i_is_random == j_is_random)) {
        //printf("Pruned for symmetry\n");
        return false;
      } // symmetry check
      */
      
      // If this ever returns false, we exit the loop because we can prune
      possibly_valid = TestHrectPair_(node_i->bound(), node_j->bound(),
                                     i, j, permutation_ok);
      
    } // for j
    
  } // for i
  
  return possibly_valid;
  
} // TestNodeTuple

void npt::SingleMatcher::BaseCaseHelper_(std::vector<std::vector<size_t> >& point_sets,
                                         std::vector<bool>& permutation_ok,
                                         std::vector<size_t>& points_in_tuple,
                                         int k) {
  
  
  std::vector<bool> permutation_ok_copy(permutation_ok);
  
  bool bad_symmetry = false;
  
  std::vector<size_t>& k_rows = point_sets[k];
  
  // iterate over possible kth members of the tuple
  for (size_t i = 0; i < k_rows.size(); i++) {
    
    size_t point_i_index = k_rows[i];
    bool this_point_works = true;
    
    bool i_is_random = (k < num_random_);
    
    bad_symmetry = false;
    
    arma::colvec vec_i;
    if (i_is_random) {
      vec_i = random_mat_.col(point_i_index);
    }
    else {
      vec_i = data_mat_.col(point_i_index);
    }
    
    // TODO: Does this leak memory?
    permutation_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
    
    // loop over points already in the tuple and check against them
    for (size_t j = 0; !bad_symmetry && this_point_works && j < k; j++) {
      
      
      bool j_is_random = (j < num_random_);
      size_t point_j_index = points_in_tuple[j];
      
      // Need to change this so it only checks if they came from the same sets
      // j comes before i in the tuple, so it should have a lower index
      //bad_symmetry = CheckSameSet_(k, j) 
      //               && (point_i_index <= point_j_index);
      bad_symmetry = (i_is_random == j_is_random) 
      && (point_i_index <= point_j_index);
      
      if (!bad_symmetry) {
        
        arma::colvec vec_j;
        if (j_is_random) {
          vec_j = random_mat_.col(point_j_index);
        }
        else {
          vec_j = data_mat_.col(point_j_index);
        }
        
        double point_dist_sq = la::DistanceSqEuclidean(vec_i, vec_j);
        
        this_point_works = TestPointPair_(point_dist_sq, j, k, 
                                          permutation_ok_copy);
        
      } // check the distances across permutations
      
    } // for j
    
    // point i fits in the tuple
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = point_i_index;
      
      // are we finished?
      if (k == tuple_size_ - 1) {
        
        results_[num_random_]++;
        
        double this_weight = 1.0;
        
        for (size_t tuple_ind = 0; tuple_ind < tuple_size_; tuple_ind++) {
          
          this_weight *= data_weights_(points_in_tuple[tuple_ind]);
          
        } // iterate over the tuple
        
        weighted_results_[num_random_] += this_weight;
        
      } 
      else {
        
        BaseCaseHelper_(point_sets, permutation_ok_copy, points_in_tuple, k+1);
        
      } // need to add more points to finish the tuple
      
    } // point i fits
    
  } // for i
  
} // BaseCaseHelper_

void npt::SingleMatcher::ComputeBaseCase(NodeTuple& nodes) {
  
  std::vector<std::vector<size_t> > point_sets(tuple_size_);
  
  // TODO: can this be done more efficiently?
  
  // Make a 2D array of the points in the nodes 
  // iterate over nodes
  for (size_t node_ind = 0; node_ind < tuple_size_; node_ind++) {
    
    //printf("point_sets[%d].size() = %d\n", node_ind, point_sets[node_ind].size());
    
    point_sets[node_ind].resize(nodes.node_list(node_ind)->count());
    
    //printf("point_sets[%d].size() = %d\n", node_ind, point_sets[node_ind].size());
    
    // fill in points in the node
    for (size_t i = 0; i < nodes.node_list(node_ind)->count(); i++) {
      
      point_sets[node_ind][i] = i + nodes.node_list(node_ind)->begin();
      
    } // points
    
  } // nodes
  
  std::vector<bool> permutation_ok(num_permutations_, true);
  
  std::vector<size_t> points_in_tuple(tuple_size_, -1);
  
  BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 0);
  
  
  
} // BaseCase

void npt::SingleMatcher::OutputResults() {
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  for (int i = 0; i <= tuple_size_; i++) {
    
    // i is the number of random points in the tuple
    std::string this_string(label_string, i, tuple_size_);
    mlpack::Log::Info << this_string << ": ";
    
    mlpack::Log::Info << results_[i] << "\n";
    
    //mlpack::Log::Info << "\n\n";
    
  } // for i
  
  
} // OutputResults
