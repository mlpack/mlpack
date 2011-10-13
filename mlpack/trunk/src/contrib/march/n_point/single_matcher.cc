/*
 *  single_matcher.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "single_matcher.h"

bool npt::SingleMatcher::CheckDistances_(double dist_sq, int ind1, 
                                         int ind2) {
  
  return (dist_sq <= upper_bounds_sqr_(ind1, ind2) && 
          dist_sq >= lower_bounds_sqr_(ind1, ind2));
  
} //CheckDistances_


// note that this assumes that the points have been checked for symmetry
bool npt::SingleMatcher::TestPointPair_(double dist_sq, int tuple_ind_1, 
                                        int tuple_ind_2,
                                        std::vector<bool>& permutation_ok) {

  bool any_matches = false;
  
  // iterate over all the permutations
  for (int i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    int template_index_1 = GetPermIndex_(i, tuple_ind_1);
    int template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
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
bool npt::SingleMatcher::TestHrectPair_(const mlpack::bound::HRectBound<2>& box1, 
                                        const mlpack::bound::HRectBound<2>& box2,
                                        int tuple_ind_1, int tuple_ind_2,
                                        std::vector<bool>& permutation_ok) {
  
  bool any_matches = false;
  
  double max_dist_sq = box1.MaxDistance(box2);
  double min_dist_sq = box1.MinDistance(box2);
  
  // iterate over all the permutations
  // Note that we have to go through all of them 
  for (int i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    int template_index_1 = GetPermIndex_(i, tuple_ind_1);
    int template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
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
  for (int i = 0; possibly_valid && i < tuple_size_; i++) {
    
    NptNode* node_i = nodes.node_list(i);
    
    // iterate over all nodes > i
    for (int j = i+1; possibly_valid && j < tuple_size_; j++) {

      NptNode* node_j = nodes.node_list(j);
      
      // If this ever returns false, we exit the loop because we can prune
      possibly_valid = TestHrectPair_(node_i->bound(), node_j->bound(),
                                     i, j, permutation_ok);
      
    } // for j
    
  } // for i
  
  return possibly_valid;
  
} // TestNodeTuple

void npt::SingleMatcher::BaseCaseHelper_(NodeTuple& nodes,
                                         std::vector<bool>& permutation_ok,
                                         std::vector<int>& points_in_tuple,
                                         int k) {
  
  
  std::vector<bool> permutation_ok_copy(permutation_ok);
  
  bool bad_symmetry = false;
  
  NptNode* kth_node = nodes.node_list(k);
  
  // iterate over possible kth members of the tuple
  for (unsigned int new_point_index = kth_node->begin(); 
       new_point_index < kth_node->end(); new_point_index++) {
    
    bool this_point_works = true;
    
    bad_symmetry = false;
    
    // we're dealing with the kth member of the tuple
    arma::colvec new_point_vec = data_mat_list_[k]->col(new_point_index);
    
    // TODO: Does this leak memory?
    permutation_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
    
    // loop over points already in the tuple and check against them
    for (int j = 0; !bad_symmetry && this_point_works && j < k; j++) {
      
      
      NptNode* old_node = nodes.node_list(j);
      
      unsigned int old_point_index = points_in_tuple[j];
      
      // TODO: is there a better way?
      // AND if they're from the same set
      bad_symmetry = (old_node == kth_node) 
                     && (new_point_index <= old_point_index);
      
      if (!bad_symmetry) {
        
        arma::colvec old_point_vec = data_mat_list_[j]->col(old_point_index);
        
        double point_dist_sq = mlpack::kernel::LMetric<2>::Evaluate(new_point_vec, 
                                                                    old_point_vec);
        
        this_point_works = TestPointPair_(point_dist_sq, j, k, 
                                          permutation_ok_copy);
        
      } // check the distances across permutations
      
    } // for j
    
    // point i fits in the tuple
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = new_point_index;
      
      // are we finished?
      if (k == tuple_size_ - 1) {
        
        
        //IMPORTANT: this is only for debugging two point
        
        //arma::colvec vec0 = data_mat_list_[0]->col(points_in_tuple[0]);
        //arma::colvec vec1 = data_mat_list_[1]->col(points_in_tuple[1]);
        
        //double dist_sq = mlpack::kernel::LMetric<2>::Evaluate(vec0, vec1);
        

        /*
        printf("tuple: (%u, %u), distance: %g\n", 
               (*old_from_new_list_[0])[points_in_tuple[0]],
               (*old_from_new_list_[1])[points_in_tuple[1]],
               sqrt(dist_sq));
        printf("%g <= %g <= %g\n\n", lower_bounds_sqr_(0,1), dist_sq, 
               upper_bounds_sqr_(0,1));
         */
        //printf("original ids: %d, %d\n", points_in_tuple[0], points_in_tuple[1]);
        
        results_++;
          
        double this_weight = 1.0;
        
        for (int tuple_ind = 0; tuple_ind < tuple_size_; tuple_ind++) {
          
          this_weight *= (*data_weights_list_[tuple_ind])[points_in_tuple[tuple_ind]];
          
        } // iterate over the tuple
        
        weighted_results_ += this_weight;
        
      } 
      else {
        
        BaseCaseHelper_(nodes, permutation_ok_copy, points_in_tuple, k+1);
        
      } // need to add more points to finish the tuple
      
    } // point i fits
    
  } // for i
  
} // BaseCaseHelper_

void npt::SingleMatcher::ComputeBaseCase(NodeTuple& nodes) {
  
  std::vector<bool> permutation_ok(num_permutations_, true);
  
  std::vector<int> points_in_tuple(tuple_size_, -1);
  
  //BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 0);
  BaseCaseHelper_(nodes, permutation_ok, points_in_tuple, 0);
  
  
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
    
    mlpack::Log::Info << results_ << "\n";
    
    //mlpack::Log::Info << "\n\n";
    
  } // for i
  
  
} // OutputResults
