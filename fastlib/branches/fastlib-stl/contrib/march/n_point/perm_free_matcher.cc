/*
 *  perm_free_matcher.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "perm_free_matcher.h"



bool npt::PermFreeMatcher::TestNodeTuple(NodeTuple& nodes) {
  
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
  
  
  for (index_t i = 0; i < upper_bounds_sqr_.size(); i++) {
    
    if (node_lower[i] > upper_bounds_sqr_[i]) {
      return false;
    }

    
    if (node_upper[i] < lower_bounds_sqr_[i]) {
      return false;
    }
     
  } // for i
  

  return true;
  
} // TestNodeTuple




bool npt::PermFreeMatcher::TestPointPair(double dist_sq, index_t tuple_index_1, 
                                         index_t tuple_index_2, 
                                         std::vector<bool>& permutation_ok) {
  
  
  bool any_matches = false;
  
  for (index_t i = 0; i < perms_.num_permutations(); i++) {
    
    if (!(permutation_ok[i])) {
      continue;
    } // does this permutation work?
    
    index_t template_index_1 = GetPermutationIndex_(i, tuple_index_1);
    index_t template_index_2 = GetPermutationIndex_(i, tuple_index_2);
    
    
    if (dist_sq <= upper_bounds_sqr_mat_(template_index_1, template_index_2) &&
        dist_sq >= lower_bounds_sqr_mat_(template_index_1, template_index_2)) {
      any_matches = true;
    } // this placement works
    else {
      permutation_ok[i] = false;
    } // this placement doesn't work
    
  } // for i
  
  return any_matches;  
  
  
} // TestPointPair



void npt::PermFreeMatcher::BaseCaseHelper_(std::vector<std::vector<index_t> >& point_sets,
                     std::vector<bool>& permutation_ok,
                     std::vector<index_t>& points_in_tuple,
                                           int k) {
  
  std::vector<bool> permutation_ok_copy(permutation_ok);
  
  bool bad_symmetry = false;
  
  std::vector<index_t>& k_rows = point_sets[k];
  
  // iterate over possible kth members of the tuple
  for (index_t i = 0; i < k_rows.size(); i++) {
    
    index_t point_i_index = k_rows[i];
    bool this_point_works = true;
    
    bad_symmetry = false;
    
    bool i_is_random = (k < num_random_);
    
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
    for (index_t j = 0; !bad_symmetry && this_point_works && j < k; j++) {
      
      index_t point_j_index = points_in_tuple[j];
      
      // j comes before i in the tuple, so it should have a lower index
      bool j_is_random = (j < num_random_);
      
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
        
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k, 
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
        
        for (int tuple_ind = 0; tuple_ind < num_random_; tuple_ind++) {
          this_weight *= random_weights_(points_in_tuple[tuple_ind]);
        }
        for (index_t tuple_ind = num_random_; tuple_ind < tuple_size_; 
             tuple_ind++) {
          
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



void npt::PermFreeMatcher::BaseCase(NodeTuple& nodes) {
  
  std::vector<std::vector<index_t> > point_sets(tuple_size_);
  
  // TODO: can this be done more efficiently?
  
  // Make a 2D array of the points in the nodes 
  // iterate over nodes
  for (index_t node_ind = 0; node_ind < tuple_size_; node_ind++) {
    
    point_sets[node_ind].resize(nodes.node_list(node_ind)->count());
    
    // fill in poilnts in the node
    /*
     for (index_t point_ind = nodes.node_list(node_ind)->begin(); 
     point_ind < nodes.node_list(node_ind)->end(); point_ind++) {
     
     point_sets[node_ind][point_ind] = point_ind;
     
     } // points
     */
    for (index_t i = 0; i < nodes.node_list(node_ind)->count(); i++) {
      
      point_sets[node_ind][i] = i + nodes.node_list(node_ind)->begin();
      
    }
    
  } // nodes
  
  std::vector<bool> permutation_ok(num_permutations_, true);
  
  std::vector<index_t> points_in_tuple(tuple_size_, -1);
  
  BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 0);
  
  
} // Base Case

