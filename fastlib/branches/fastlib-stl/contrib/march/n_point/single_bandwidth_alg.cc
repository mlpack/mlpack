/*
 *  single_bandwidth_alg.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "single_bandwidth_alg.h"

// Returns true if ind1 and ind2 in the tuple are both data or both random
/*
bool npt::SingleBandwidthAlg::CheckSameSet_(int ind1, int ind2) {
  
  if (ind1 > ind2) {
    std::swap(ind1,ind2);
  }
  
  if (ind2 <= num_random_) {
    return true;
  }
  else if (ind1 > num_random_) {
    return true;
  }
  else {
    return false;
  }
  
}
 */

bool npt::SingleBandwidthAlg::CheckNodeList_(std::vector<SingleNode*>& nodes) {
  
  bool can_prune = false;
  
  // note that this has to enforce symmetry
  std::vector<bool> permutation_ok(matcher_.num_permutations(), true);
  
  // iterate over all nodes
  // IMPORTANT: right now, I'm exiting when I can prune
  // I need to double check that this works
  for (index_t i = 0; !can_prune && i < tuple_size_; i++) {
    
    bool i_is_random = i < num_random_;
    
    SingleNode* node_i = nodes[i];
    
    // iterate over all nodes > i
    for (index_t j = i+1; !can_prune && j < tuple_size_; j++) {
      
      bool j_is_random = j < num_random_;
      
      SingleNode* node_j = nodes[j];
      
      // check for symmetry
      if (node_j->end() <= node_i->begin() && (i_is_random == j_is_random)) {
        //printf("Pruned for symmetry\n");
        return true;
      } // symmetry check
      
      // IMPORTANT: need to negate this, TestHrectPair returns true if the 
      // pair may contain tuples
      can_prune = !(matcher_.TestHrectPair(node_i->bound(), node_j->bound(),
                                           i, j, permutation_ok));
      
    } // for j
    
  } // for i
  
  return can_prune;
  
} // CheckNodeList



void npt::SingleBandwidthAlg::BaseCaseHelper_(std::vector<std::vector<index_t> >& point_sets,
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
    
    bool i_is_random = (k < num_random_);
    
    bad_symmetry = false;
    
    arma::colvec vec_i;
    if (i_is_random) {
      vec_i = random_points_.col(point_i_index);
    }
    else {
      vec_i = data_points_.col(point_i_index);
    }
    
    // TODO: Does this leak memory?
    permutation_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
    
    // loop over points already in the tuple and check against them
    for (index_t j = 0; !bad_symmetry && this_point_works && j < k; j++) {
      

      bool j_is_random = (j < num_random_);
      index_t point_j_index = points_in_tuple[j];
      
      // Need to change this so it only checks if they came from the same sets
      // j comes before i in the tuple, so it should have a lower index
      //bad_symmetry = CheckSameSet_(k, j) 
      //               && (point_i_index <= point_j_index);
      bad_symmetry = (i_is_random == j_is_random) 
                       && (point_i_index <= point_j_index);
      
      if (!bad_symmetry) {
        
        arma::colvec vec_j;
        if (j_is_random) {
          vec_j = random_points_.col(point_j_index);
        }
        else {
          vec_j = data_points_.col(point_j_index);
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
        
        num_tuples_[num_random_]++;
        
        double this_weight = 1.0;
        
        for (index_t tuple_ind = 0; tuple_ind < tuple_size_; tuple_ind++) {
          
          this_weight *= data_weights_(points_in_tuple[tuple_ind]);
          
        } // iterate over the tuple
        
        weighted_num_tuples_[num_random_] += this_weight;
        
      } 
      else {
        
        BaseCaseHelper_(point_sets, permutation_ok_copy, points_in_tuple, k+1);
        
      } // need to add more points to finish the tuple
      
    } // point i fits
    
  } // for i
  
} // BaseCaseHelper_



void npt::SingleBandwidthAlg::BaseCase_(std::vector<SingleNode*>& nodes) {
  
  std::vector<std::vector<index_t> > point_sets(tuple_size_);

  // TODO: can this be done more efficiently?
  
  // Make a 2D array of the points in the nodes 
  // iterate over nodes
  for (index_t node_ind = 0; node_ind < tuple_size_; node_ind++) {
    
    //printf("point_sets[%d].size() = %d\n", node_ind, point_sets[node_ind].size());
    
    point_sets[node_ind].resize(nodes[node_ind]->count());
    
    //printf("point_sets[%d].size() = %d\n", node_ind, point_sets[node_ind].size());
    
    // fill in points in the node
    for (index_t i = 0; i < nodes[node_ind]->count(); i++) {
      
      point_sets[node_ind][i] = i + nodes[node_ind]->begin();
      
    } // points
    
  } // nodes
  
  std::vector<bool> permutation_ok(matcher_.num_permutations(), true);
  
  std::vector<index_t> points_in_tuple(tuple_size_, -1);
  
  BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 0);
  
} // BaseCase_()




void npt::SingleBandwidthAlg::DepthFirstRecursion_(std::vector<SingleNode*>& nodes) {
  
  // check for symmetry and pruning
  bool can_prune = CheckNodeList_(nodes);
  
  if (can_prune) {
    
    //std::cout << "Can prune\n";
    
    // note that this will count prunes based on symmetry too
    num_prunes_++;
    
  }
  else {
    
    //std::cout << "Can't prune\n";
    
    // look over all the nodes, see if they are leaves, and if not, which one 
    // to split 
    bool all_leaves = nodes[0]->is_leaf();
    index_t split_index = 0;
    // if node 0 is not a leaf, then use it, otherwise don't
    index_t split_count = all_leaves ? 0 : nodes[0]->count();

    // loop over the other nodes, check for leaves and who to split
    for (index_t i = 1; i < tuple_size_; i++) {
     
      if (!(nodes[i]->is_leaf())) {
        
        all_leaves = false;
        
        if (nodes[i]->count() > split_count) {
          split_count = nodes[i]->count();
          split_index = i;
        }
        
      } // not a leaf
      
    } // for i
    
    // Do we do the base case or recurse?
    
    if (all_leaves) {
      
      BaseCase_(nodes);
      num_base_cases_++;
      
    } 
    else {
      
      SingleNode* split_node = nodes[split_index];
      
      // left child first
      nodes[split_index] = split_node->left();
      DepthFirstRecursion_(nodes);
      
      // now do the other child
      nodes[split_index] = split_node->right();
      DepthFirstRecursion_(nodes);
      
      // This is important for calls above this one.
      nodes[split_index] = split_node;
      
      
    } // not a base case
          
  } // can't prune
  
} // DepthFirstRecursion_
