/*
 *  n_point.cc
 *  
 *
 *  Created by William March on 2/24/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point.h"

// returns true if the indices violate the symmetry requirement
bool NPointAlg::PointsViolateSymmetry_(index_t ind1, index_t ind2) {
  return (ind2 <= ind1);
} // PointsViolateSymmetry_()

int NPointAlg::BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                                  ArrayList<bool>& permutation_ok,
                                  ArrayList<index_t>& points_in_tuple,
                                  double* weighted_result, int k) {
  
  int result = 0;
  
  ArrayList<bool> permutation_ok_copy;
  permutation_ok_copy.InitCopy(permutation_ok);
  
  ArrayList<index_t> k_rows;
  k_rows.InitAlias(point_sets[k]);
  
  bool bad_symmetry = false;
  
  // loop over possible points for the kth member of the tuple
  for (index_t i = 0; !bad_symmetry && i < k_rows.size(); i++) {
    
    index_t point_index_i = k_rows[i];
    
    bool this_point_works = true;

    Vector point_i;
    data_points_.MakeColumnVector(point_index_i, &point_i);
    
    // TODO: is this too inefficient?
    permutation_ok_copy.Clear();
    permutation_ok_copy.InitCopy(permutation_ok);
    
    // loop over previously chosen points to see if there is a conflict
    for (index_t j = 0; !bad_symmetry && this_point_works && j < k; j++) {
      
      index_t point_index_j = points_in_tuple[j];
      
      bad_symmetry = PointsViolateSymmetry_(point_index_i, point_index_j);
      
      // don't compute the distances if we don't have to
      if (!bad_symmetry) {
        Vector point_j;
        data_points_.MakeColumnVector(point_index_j, &point_j);
        
        double point_dist_sq = la::DistanceSqEuclidean(point_i, point_j);
        
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k, 
                                                  permutation_ok_copy);
        
      }
      
    } // for j
    
    // now, if the point passed, we put it in place and recurse
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = point_index_i;
      
      // base case of the recursion
      if (k == num_tuples_ - 1) {
        
        result++;
        
        // take care of weights
        double these_weights = 1.0;
        
        for (index_t a = 0; a < num_tuples_; a++) {
          these_weights *= data_weights_[points_in_tuple[a]];
        } // for a
        
        (*weighted_result) = (*weighted_result) + these_weights;
        
      } // base case
      else {
        
        result += BaseCaseHelper_(point_sets, permutation_ok_copy, 
                                  points_in_tuple, weighted_result, k+1);
        
      } // recurse
      
      DEBUG_ONLY(points_in_tuple[k] = -1);
      
    } // did the point work
    
  } // for i
  
  return result;
  
} // BaseCaseHelper_()


int NPointAlg::BaseCase_(ArrayList<ArrayList<index_t> >& point_sets, 
                         double* weighted_result) {

  ArrayList<bool> permutation_ok;
  permutation_ok.Init(matcher_.num_permutations());
  for (index_t i = 0; i < matcher_.num_permutations(); i++) {
    permutation_ok[i] = true;
  } // for i
  
  ArrayList<index_t> points_in_tuple;
  points_in_tuple.Init(tuple_size_);
  
#ifdef DEBUG
  for (index_t j = 0; j < tuple_size_; j++) {
    points_in_tuple[j] = -1;
  } // for j
#endif
  
  int result = BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 
                               weighted_result, 0);
  
  return result;
  
} // BaseCase_()


