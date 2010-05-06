/*
 *  n_point_perm_free.cc
 *  
 *
 *  Created by William March on 4/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_perm_free.h"

// returns true if the indices violate the symmetry requirement
bool NPointPermFree::PointsViolateSymmetry_(index_t ind1, index_t ind2) {
  DEBUG_ASSERT(ind1 >= 0);
  DEBUG_ASSERT(ind2 >= 0);
  return (ind2 <= ind1);
} // PointsViolateSymmetry_()


int NPointPermFree::BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                                    ArrayList<bool>& permutation_ok,
                                    ArrayList<index_t>& points_in_tuple, 
                                    int k) {
  
  
  int result = 0;
  
  ArrayList<bool> permutation_ok_copy;
  permutation_ok_copy.InitCopy(permutation_ok);
  
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
    
    
    // TODO: figure out a way to handle the bad symmetry more elegantly
    // I can't just exit the loop when the symmetry is bad, but I also can't
    
    
    // loop over previously chosen points to see if there is a conflict
    // IMPORTANT: I'm assuming here that once a point violates symmetry,
    // all the other points will too
    // This means I'm assuming that the lists of points are in a continuous 
    // order
    //for (index_t j = 0; !bad_symmetry && this_point_works && j < k; j++) {
    for (index_t j = 0; this_point_works && j < k; j++) {
      
      index_t point_index_j = points_in_tuple[j];
      
      // need to swap indices here, j should come before i
      bad_symmetry = PointsViolateSymmetry_(point_index_j, point_index_i);
      //printf("point_j: %d, point_i: %d, bad_symmetry: %d\n", point_index_j, 
      //       point_index_i, bad_symmetry);
      
      // don't compute the distances if we don't have to
      if (!bad_symmetry) {
        Vector point_j;
        data_points_.MakeColumnVector(point_index_j, &point_j);
        
        double point_dist_sq = la::DistanceSqEuclidean(point_i, point_j);
        
        //printf("Testing point pair (%d, %d)\n", j, k);
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k, 
                                                  permutation_ok_copy);
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
        
        result++;
        
      } // base case
      else {
        
        result += BaseCaseHelper_(point_sets, permutation_ok_copy, 
                                  points_in_tuple, k+1);
        
      } // recurse
      
      DEBUG_ONLY(points_in_tuple[k] = -1);
      
      points_in_tuple[k] = -1;
      
    } // did the point work
    
  } // for i
  
  return result;
  
} // BaseCaseHelper_()


int NPointPermFree::BaseCase_(NodeTuple& nodes) {
  
  ArrayList<ArrayList<index_t> > point_sets;
  point_sets.Init(tuple_size_);
  
  for (index_t i = 0; i < tuple_size_; i++) {
    point_sets[i].Init(nodes.node_list(i)->count());
    
    for (index_t j = 0; j < nodes.node_list(i)->count(); j++) {
      point_sets[i][j] = j + nodes.node_list(i)->begin();
    } // for j
    
  } // for i
  
  
  // form lists of points, use same base case from before
  ArrayList<bool> permutation_ok;
  permutation_ok.InitRepeat(true, matcher_.num_permutations());
  
  ArrayList<index_t> points_in_tuple;
  points_in_tuple.InitRepeat(-1, tuple_size_);
  
  int result = BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 0);
  
  return result;
  
} // BaseCase_()

// TODO: add subsumes
bool NPointPermFree::CanPrune_(NodeTuple& nodes) {
  
  // check the matcher
  if (matcher_.CheckNodes(nodes) == EXCLUDE) {
    return true;
  }
  else {
    return false;
  }
  
} // CanPrune_()

int NPointPermFree::DepthFirstRecursion_(NodeTuple& nodes) {
  
  int num_tuples_here = 0;
  
  if (CanPrune_(nodes)) {

    printf("Pruning.\n");
    // do it
    num_exclusion_prunes_++;
    return num_tuples_here;
  
  }
  else if (nodes.all_leaves()) {
    
    printf("Performing Base Case.\n");
    int base_case_count = BaseCase_(nodes);
    printf("Num points in base case: %d\n", base_case_count);
    num_tuples_here = base_case_count;
    
  }// base case
  else {
   
    // perform split
    // check symmetry
    // check prunes
    
    printf("Recursing.\n");
    
    NodeTuple left_node;
    NodeTuple* left_node_ptr = &left_node;
    
    NodeTuple right_node;
    NodeTuple* right_node_ptr = &right_node;
    
    nodes.PerformSplit(left_node_ptr, right_node_ptr);
    
    // check if the list of bandwidths is still sorted here
    
    
    if (left_node_ptr) {
      
      printf("Left node\n");
      left_node.Print();
      DEBUG_ASSERT(left_node.node_list(0));
      num_tuples_here += DepthFirstRecursion_(left_node);
    }
    if (right_node_ptr) {
      
      printf("Right node\n");
      right_node.Print();
      DEBUG_ASSERT(right_node.node_list(0));
      num_tuples_here += DepthFirstRecursion_(right_node);
    }
    
    
    
  } // not base case
  
  return num_tuples_here;
  
} // DFS
