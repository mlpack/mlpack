/*
 *  n_point_multi.cc
 *  
 *
 *  Created by William March on 4/14/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_multi.h"


bool NPointMulti::SymmetryCorrect_(ArrayList<NPointNode*>& nodes) {
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    NPointNode* node_i = nodes[i];
    
    for (index_t j = i+1; j < tuple_size_; j++) {
      
      NPointNode* node_j = nodes[j];
      
      if (node_j->end() <= node_i->begin()) {
        return false;
      }
      
    } // for j
  } // for i
  
  return true;
  
} // SymmetryCorrect_

// fills inds with the indices in the range list that need to be recomputed
void NPointMulti::FindInvalidIndices_() {
  
  invalid_indices_.Init(tuple_size_);
  
  for (index_t split_ind = 0; split_ind < tuple_size_; split_ind++) {
    
    invalid_indices_[split_ind].Init();
    
    // inserted this easy fix, not sure if the rest is right yet
    if (tuple_size_ == 2) {
      invalid_indices_[split_ind].PushBackCopy(0);
    }
    else {
      
      index_t bad_ind = split_ind - 1;
      index_t bad_ind2 = 0;
      
      for (index_t i = 0; i < split_ind; i++) {
        
        invalid_indices_[split_ind].PushBackCopy(bad_ind);
        bad_ind += tuple_size_ - 1 - (i+1);
        bad_ind2 += tuple_size_ - i - 1;
        
      } // horizontal
      
      for (index_t i = split_ind+1; i < tuple_size_; i++) {
        
        invalid_indices_[split_ind].PushBackCopy(bad_ind2);
        bad_ind2++;
        
      }
      
    } // n > 2
    
  } // for split_ind
  
} // FindInvalidIndices_()

index_t NPointMulti::CheckBaseCase_(ArrayList<NPointNode*>& nodes) {
  
  index_t split_ind = -1;
  int split_size = 0;
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    if (!(nodes[i]->is_leaf())) {
      
      if (nodes[i]->count() > split_size) {
        split_ind = i;
        split_size = nodes[i]->count();
      }
      
    }
    
  } // for i
  
  return split_ind;
  
} // CheckBaseCase_()

// needs to find the right index among (n choose 2) quantities, where i is 
// less than j
index_t NPointMulti::FindInd_(index_t i, index_t j) {
  
  DEBUG_ASSERT(i < j);
  
  return ((i - 1) + ((j-1)*(j-2)/2));
  
} // FindInd_()

// returns true if the indices violate the symmetry requirement
bool NPointMulti::PointsViolateSymmetry_(index_t ind1, index_t ind2) {
  DEBUG_ASSERT(ind1 >= 0);
  DEBUG_ASSERT(ind2 >= 0);
  return (ind2 <= ind1);
} // PointsViolateSymmetry_()



void NPointMulti::BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                                  ArrayList<bool>& permutation_ok,
                                  ArrayList<index_t>& points_in_tuple, 
                                  int k, 
                                  ArrayList<GenMatrix<index_t> >& permutation_ranges) {
  
  ArrayList<bool> permutation_ok_copy;
  permutation_ok_copy.InitCopy(permutation_ok);
  
  ArrayList<GenMatrix<index_t> > permutation_ranges_copy;
  permutation_ranges_copy.InitCopy(permutation_ranges);
  
  ArrayList<index_t> k_rows;
  k_rows.InitAlias(point_sets[k]);
  
  bool bad_symmetry = false;
  
  // loop over possible points for the kth member of the tuple
  //for (index_t i = 0; !bad_symmetry && i < k_rows.size(); i++) {
  // IMPORTANT: can't exit here for bad symmetry, it can get better as 
  // i increases
  for (index_t i = 0; i < k_rows.size(); i++) {
    
    index_t point_index_i = k_rows[i];
    
    bool this_point_works = true;
    
    Vector point_i;
    data_points_.MakeColumnVector(point_index_i, &point_i);
    
    // TODO: is this too inefficient?
    permutation_ok_copy.Clear();
    permutation_ok_copy.AppendCopy(permutation_ok);

    permutation_ranges_copy.Clear();
    permutation_ranges_copy.AppendCopy(permutation_ranges);

    
    // TODO: figure out a way to handle the bad symmetry more elegantly
    // I should be able to avoid a bit more work
    for (index_t j = 0; this_point_works && j < k; j++) {
      
      index_t point_index_j = points_in_tuple[j];
      
      // j should come before i since j comes first 
      bad_symmetry = PointsViolateSymmetry_(point_index_j, point_index_i);
      //printf("point_j: %d, point_i: %d, bad_symmetry: %d\n", point_index_j, 
      //       point_index_i, bad_symmetry);
      
      // don't compute the distances if we don't have to
      if (!bad_symmetry) {
        Vector point_j;
        data_points_.MakeColumnVector(point_index_j, &point_j);
        
        double point_dist_sq = la::DistanceSqEuclidean(point_i, point_j);
        
        //printf("Testing point pair (%d, %d)\n", point_index_j, 
        //       point_index_i);
        // This needs to fill in the permutation_ok_copy for each matcher
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k, 
                                                  permutation_ok_copy, 
                                                  permutation_ranges_copy);
        //printf("this_point_works: %d\n", this_point_works);
        
      } // compute the distances and check the matcher
      
    } // for j
    
    /*
     printf("Considering point %d in position %d.  bad_symmetry: %d, works: %d\n",
     point_index_i, k, bad_symmetry, this_point_works);
     */
    
    // now, if the point passed, we put it in place and recurse
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = point_index_i;
      
      // base case of the recursion
      if (k == tuple_size_ - 1) {
        
        results_.ClearFilledResults();
        
        //ot::Print(permutation_ok_copy);
        
        for (index_t perm_index = 0; perm_index < matcher_.num_permutations(); 
             perm_index++) {
          
          // this one won't fit anywhere anyway
          if (! permutation_ok_copy[perm_index]) {
            continue;
          }
          
          //printf("perm_index: %d\n", perm_index);
          //permutation_ranges_copy[perm_index].PrintDebug("PermutationRangesCopy");
          //ot::Print(permutation_ranges_copy[perm_index]);
          
          results_.IncrementRange(permutation_ranges_copy[perm_index]);
          
          
        } // iterate over permutations
        
      } // base case
      else {
        
        BaseCaseHelper_(point_sets, permutation_ok_copy,  
                        points_in_tuple, k+1, permutation_ranges_copy);
        
      } // recurse
      
      //DEBUG_ONLY(points_in_tuple[k] = -1);
      
    } // did the point work
    
  } // for i
  
  
} // BaseCaseHelper_()



// Collect the indices of the valid matchers, check against each one?
// How to re-use info?  
void NPointMulti::BaseCase_(NodeTuple& nodes, 
                            ArrayList<std::pair<double, double> >& valid_ranges) {
  
  // Create the lists of points
  ArrayList<ArrayList<index_t> > point_sets;
  point_sets.Init(tuple_size_);
  
  for (index_t i = 0; i < tuple_size_; i++) {
    point_sets[i].Init(nodes.node_list(i)->count());
    
    for (index_t j = 0; j < nodes.node_list(i)->count(); j++) {
      point_sets[i][j] = j + nodes.node_list(i)->begin();
    } // for j
    
  } // for i
  
  
  ArrayList<GenMatrix<index_t> > permutation_ranges;
  permutation_ranges.Init(matcher_.num_permutations());
  
  for (int i = 0; i < permutation_ranges.size(); i++) {
    
    permutation_ranges[i].Init(tuple_size_, tuple_size_);
    // TODO: do I need to initialize it to some safe value?
    
  } // fill in permutation matrices
  
  
  ArrayList<bool> permutation_ok;
  permutation_ok.InitRepeat(true, matcher_.num_permutations());

  ArrayList<index_t> points_in_tuple;
  points_in_tuple.InitRepeat(-1, tuple_size_);
  
  // TODO: figure out which matchers we need to worry about here
  
  BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 0, 
                  permutation_ranges);
  
} // BaseCase_()


// valid_ranges are the ranges of indices in the distances_ array in the matcher
// it has length (n choose 2), the lower ends should be strictly non-decreasing
void NPointMulti::DepthFirstRecursion_(NodeTuple& nodes, 
                                       ArrayList<std::pair<double, double> >& valid_ranges) {
  
  bool can_prune = false;
  
  // update valid_ranges
  // valid_ranges holds the range of distances that WON'T prune
  // i.e. the only matchers that can't be pruned are ones that have a non-empty
  // overlap with valid_ranges[i] for all i
  for (index_t i = 0; i < valid_ranges.size(); i++) {

    // IMPORTANT: first is lo, second is hi
    valid_ranges[i].second = min(valid_ranges[i].second, nodes.upper_bound(i));
    valid_ranges[i].first = max(valid_ranges[i].first, nodes.lower_bound(i));
    
    /*
    if (valid_ranges[i].first >= valid_ranges[i].second) {
      printf("Pruning on empty range.\n");
      can_prune = true;
      break;
    } // check if the range is empty
    */
    // TODO: make sure that it's not too small or large for any matcher
    
    if (valid_ranges[i].first > matcher_.max_dist()) {
      //printf("Pruning on too much separation for any matcher.\n");
      can_prune = true;
      break;
    } // too large
     
    
    // add lower bounds here later 
    
  } // update ranges
  
  // check prune - i.e. check if it's still possible to contribute to anything
  if (can_prune) {
    //printf("Pruned all\n");
    num_total_prunes_++;
    return;
  } // check prune
  else if (nodes.all_leaves()) {
    //printf("Base Case\n");
    BaseCase_(nodes, valid_ranges);
  } // base case
  else {
    
    NodeTuple left_node;
    NodeTuple* left_node_ptr = &left_node;
    
    NodeTuple right_node;
    NodeTuple* right_node_ptr = &right_node;
    
    // just pass in the invalid indices here
    nodes.PerformSplit(left_node_ptr, right_node_ptr, invalid_indices_);
    
    // check if the list of bandwidths is still sorted here
    
    
    if (left_node_ptr) {
      
      //printf("Left node\n");
      //left_node.Print();
      ArrayList<std::pair<double, double> > left_ranges;
      left_ranges.InitCopy(valid_ranges);
      DEBUG_ASSERT(left_node.node_list(0));
      DepthFirstRecursion_(left_node, left_ranges);
    
    }
    if (right_node_ptr) {
      
      ArrayList<std::pair<double, double> > right_ranges;
      right_ranges.InitCopy(valid_ranges);
      //printf("Right node\n");
      //right_node.Print();
      DEBUG_ASSERT(right_node.node_list(0));
      DepthFirstRecursion_(right_node, right_ranges);
      
    }
    
  } // recurse
  
} // DepthFirstRecursion_()




