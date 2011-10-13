/*
 *  multi_bandwidth_alg.cc
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "multi_bandwidth_alg.h"


size_t npt::MultiBandwidthAlg::FindResultsInd_(
               const std::vector<size_t>& perm_locations) {
  
  //std::cout << "Finding results ind\n";
  
  size_t result = 0;
  size_t num_previous_bands = 1;
  
  for (size_t i = 0; i < perm_locations.size(); i++) {
   
    result += perm_locations[i] * num_previous_bands;
    num_previous_bands *= num_bands_[i];
    
  }
  
  return result;
  
} // FindResultsInd

// this is the inverse of the function above
void npt::MultiBandwidthAlg::FindMatcherInd_(size_t loc,
                                             std::vector<size_t>& result) {
  
  //std::vector<size_t> result(num_bands_.size());

  size_t new_loc = loc;

  size_t mod_fac = 1;
  
  for (size_t i = 0; i < num_bands_.size(); i++) {
    mod_fac *= num_bands_[i];
  }
  
  
  for (int i = num_bands_.size() - 1; i >= 0; i--) {
    
    result[i] = new_loc;
    
    mod_fac = mod_fac / num_bands_[i];
    
    new_loc = new_loc % mod_fac;
    
  } // for i
  

  mod_fac = num_bands_[0];
  
  for (size_t i = 1; i < result.size(); i++) {
    
    for (size_t j = 0; j < i; j++) {
    
      result[i] = result[i] - result[j];
    
    } // for j
    
    result[i] = result[i] / mod_fac;
    
    mod_fac = mod_fac * num_bands_[i];
    
  } // for i
  
} // FindMatcherInd


void npt::MultiBandwidthAlg::BaseCaseHelper_(
                         std::vector<std::vector<size_t> >& point_sets,
                         std::vector<bool>& permutation_ok,
                         std::vector<std::vector<size_t> >& perm_locations,
                         std::vector<size_t>& points_in_tuple,
                         int k) {
  
  
  // perm_locations[i][j] = k means that in the ith permutation, that 
  // matcher_dists_[j][k] is the current entry in the matcher that this tuple
  // satisfies
  
  std::vector<bool> perm_ok_copy(permutation_ok);
  std::vector<std::vector<size_t> > perm_locations_copy(perm_locations);

  bool bad_symmetry = false;
  
  // iterate over possible new points
  for (size_t i = 0; i < point_sets[k].size(); i++) {
    
    size_t new_point_ind = point_sets[k][i];
    bool this_point_works = true;
    
    bad_symmetry = false;
    
    bool i_is_random = (k < num_random_);
    
    arma::colvec new_point_vec;
    
    if (i_is_random) {
      new_point_vec = random_points_.col(new_point_ind);
    } 
    else {
      new_point_vec = data_points_.col(new_point_ind);
    }
    
    // copy the permutation 
    perm_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
    
    // TODO: check if I can accurately copy this more directly
    for (size_t m = 0; m < perm_locations_copy.size(); m++) {
      perm_locations_copy[m].assign(perm_locations[m].begin(), 
                                    perm_locations[m].end());
    } // for m
    
    // TODO: double check that I can exit on bad symmetry here
    for (size_t j = 0; j < k && this_point_works && !bad_symmetry; j++) {
      
      size_t old_point_ind = points_in_tuple[j];
      
      bool j_is_random = (j < num_random_);
      
      bad_symmetry = (i_is_random == j_is_random) 
                      && (new_point_ind <= old_point_ind);
      
      // TODO: if bad_symmetry, can I break out of the loop?
      if (!bad_symmetry) {
        
        arma::colvec old_point_vec;
        if (j_is_random) {
          old_point_vec = random_points_.col(old_point_ind);
        }
        else {
          old_point_vec = data_points_.col(old_point_ind);
        }
        
        double point_dist_sq = la::DistanceSqEuclidean(old_point_vec, 
                                                       new_point_vec);
        
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k, 
                                                  perm_ok_copy,
                                                  perm_locations_copy);
        // perm_locations_copy should now be filled in 
        
      } // check symmetry
      
    } // check existing points
    
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = new_point_ind;
      
      if (k == tuple_size_ - 1) {
        
        // fill in all the results that worked
        
        std::set<size_t> results_set;
        
        for (size_t n = 0; n < perm_locations_copy.size(); n++) {
          
          if (perm_ok_copy[n]) {
            size_t results_ind = FindResultsInd_(perm_locations_copy[n]);
            results_set.insert(results_ind);
            //std::cout << "Inserting: " << results_ind << "\n";
          }
        } // for n
        
        // Now, iterate through all (distinct) results keys in the set and add
        // them to the total
        std::set<size_t>::iterator it;
        
        for (it = results_set.begin(); it != results_set.end(); it++) {
          
          results_[num_random_][*it]++;
          
        }
        
        double this_weight = 1.0;
        for (int tuple_ind = 0; tuple_ind < num_random_; tuple_ind++) {
          this_weight *= random_weights_(points_in_tuple[tuple_ind]);
        }
        for (size_t tuple_ind = num_random_; tuple_ind < tuple_size_; 
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

void npt::MultiBandwidthAlg::BaseCase_(NodeTuple& nodes) {
  
  std::vector<std::vector<size_t> > point_sets(tuple_size_);
  
  for (size_t node_ind = 0; node_ind < tuple_size_; node_ind++) {
    
    point_sets[node_ind].resize(nodes.node_list(node_ind)->count());
    
    for (size_t i = 0; i < nodes.node_list(node_ind)->count(); i++) {
      
      point_sets[node_ind][i] = i + nodes.node_list(node_ind)->begin();
      
    } // for i
    
  } // for node_ind
  
  std::vector<bool> permutation_ok(matcher_.num_permutations(), true);
  
  std::vector<size_t> points_in_tuple(tuple_size_, -1);
  
  std::vector<std::vector<size_t> > perm_locations(num_permutations_);

  for (size_t i = 0; i < perm_locations.size(); i++) {
    perm_locations[i].resize(num_bands_.size(), INT_MAX);
  }
  
  BaseCaseHelper_(point_sets, permutation_ok, perm_locations, 
                  points_in_tuple, 0);
  
  
} // BaseCase_




bool npt::MultiBandwidthAlg::CanPrune_(NodeTuple& nodes) {

  return (!(matcher_.TestNodeTuple(nodes)));
  
} // CanPrune_




void npt::MultiBandwidthAlg::DepthFirstRecursion_(NodeTuple& nodes) {
  
  if (CanPrune_(nodes)) {
    num_prunes_++;
  }
  else if (nodes.all_leaves()) {
    
    num_base_cases_++;
    BaseCase_(nodes);
    
  } // base case
  else {
    
    if (nodes.CheckSymmetry(nodes.ind_to_split(), true)) {
      
      NodeTuple left_child(nodes, true);
      DepthFirstRecursion_(left_child);
      
    }
    
    if (nodes.CheckSymmetry(nodes.ind_to_split(), false)) {

      NodeTuple right_child(nodes, false);
      DepthFirstRecursion_(right_child);
      
    }
    
  } // recursing
  
} // Recursion
          
        
void npt::MultiBandwidthAlg::OutputResults() {
  
  //std::cout << "First result: " << results_[0] << "\n\n";
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  for (int i = 0; i <= tuple_size_; i++) {
    
    // i is the number of random points in the tuple
    std::string this_string(label_string, i, tuple_size_);
    mlpack::Log::Info << this_string << "\n";
    
    for (size_t j = 0; j < results_[i].size(); j++) {
      
      std::vector<size_t> matcher_ind(num_bands_.size());
      FindMatcherInd_(j, matcher_ind);
      
      mlpack::Log::Info << "Matcher: ";
      for (size_t k = 0; k < matcher_ind.size(); k++) {
        
        mlpack::Log::Info << matcher_.matcher_dists(k, matcher_ind[k]) << ", ";
        
      } // for k
      mlpack::Log::Info << ": " << results_[i][j] << "\n";
      
    } // for j
    
    mlpack::Log::Info << "\n\n";
    
  } // for i
  
  
} // OutputResults

