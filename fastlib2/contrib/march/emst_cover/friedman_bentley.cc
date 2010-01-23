/*
 *  friedman_bentley.cc
 *  
 *
 *  Created by William March on 1/4/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "friedman_bentley.h"

void FriedmanBentley::FindNeighbor_(FBTree* tree, const Vector& point, 
                                    index_t point_index,
                                    index_t* cand, double* dist) {

  if (tree->is_leaf()) {
    
    for (index_t i = tree->begin(); i < tree->end(); i++) {
      
      if (connections_.Find(i) != connections_.Find(point_index)) {
        
        Vector point_i;

        data_points_.MakeColumnVector(i, &point_i);
        double this_dist = la::DistanceSqEuclidean(point, point_i);
        
        if (this_dist < *dist) {
          *dist = this_dist;
          *cand = i;
        } // is the distance minimum?
        
      } // are they connected?
      
    } // for i
    
  } // base case
  else {
  
    double left_dist, right_dist;
    
    if (tree->left()->stat().component_membership() != 
        connections_.Find(point_index)) {
      left_dist = tree->left()->bound().MinDistanceSq(point);
    }
    else {
      left_dist = DBL_MAX;
    }
    
    if (tree->right()->stat().component_membership() != 
        connections_.Find(point_index)) {
      right_dist = tree->right()->bound().MinDistanceSq(point);
    }
    else {
      right_dist = DBL_MAX;
    }
    
    // prioritize by distance
    if (left_dist < right_dist) {
     
      if (left_dist < *dist) {
        
        FindNeighbor_(tree->left(), point, point_index, cand, dist);
        
      }
      
      if (right_dist < *dist) {

        FindNeighbor_(tree->right(), point, point_index, cand, dist);

      }
      
    } // prioritizing by distance
    else {

      if (right_dist < *dist) {

        FindNeighbor_(tree->right(), point, point_index, cand, dist);
        
      }
      
      if (left_dist < *dist) {

        FindNeighbor_(tree->left(), point, point_index, cand, dist);

      }
      
    } // prioritize distances
    
  } // not base case
  
} // FindNeighbor_

void FriedmanBentley::UpdateMemberships_(FBTree* tree, index_t point_index) {
  
  if (tree->is_leaf()) {

    index_t comp_index = connections_.Find(point_index);

    for (index_t i = tree->begin(); i < tree->end(); i++) {
     
      if (comp_index != connections_.Find(i)) {
        comp_index = -1;
        break;
      }
      
    } // iterate over base case points
    
    tree->stat().set_component_membership(comp_index);
    
  } // base case
  // go left
  else if (point_index < tree->left()->end()) {
    
    UpdateMemberships_(tree->left(), point_index);
    
    if (tree->left()->stat().component_membership() 
        == tree->right()->stat().component_membership()) {
      tree->stat().set_component_membership(tree->left()->stat().component_membership());
    }
    
  }
  else { // go right

    UpdateMemberships_(tree->right(), point_index);
    
    if (tree->left()->stat().component_membership() 
        == tree->right()->stat().component_membership()) {
      tree->stat().set_component_membership(tree->left()->stat().component_membership());
    }
    
  }
  
} // UpdateMemberships_()


void FriedmanBentley::ComputeMST(Matrix* results) {

  fx_timer_start(mod_, "MST_computation");
  
  index_t cand = -1;
  double cand_dist = DBL_MAX;
  
  Vector point;
  data_points_.MakeColumnVector(0, &point);
  
  // find the neighbor of point 0 to start the fragment
  FindNeighbor_(tree_, point, 0, &cand, &cand_dist);
  candidate_neighbors_[0] = cand;
  
  heap_.Put(cand_dist, 0);
  
  // main loop
  while (number_of_edges_ < number_of_points_ - 1) {
    
    DEBUG_ASSERT(heap_.size() == number_of_edges_ + 1);
    
    double dist = heap_.top_key();
    index_t point_in_fragment = heap_.Pop();
    index_t point_out_fragment = candidate_neighbors_[point_in_fragment];
    
    // if the link is not real
    if (connections_.Find(point_in_fragment) 
        == connections_.Find(point_out_fragment)) {
      
      Vector point_vec;
      data_points_.MakeColumnVector(point_in_fragment, &point_vec);
      
      index_t new_point = -1;
      double new_dist = DBL_MAX;
      
      FindNeighbor_(tree_, point_vec, point_in_fragment, &new_point, &new_dist);
      candidate_neighbors_[point_in_fragment] = new_point;
      
      heap_.Put(new_dist, point_in_fragment);
      
      //num_nearest_neighbor_computations_++;
      
    } 
    else { // link is real, add it
      
      AddEdge_(point_in_fragment, point_out_fragment, dist);
      connections_.Union(point_in_fragment, point_out_fragment);
      
      UpdateMemberships_(tree_, point_in_fragment);
      UpdateMemberships_(tree_, point_out_fragment);
      
      Vector point_out_vec;
      data_points_.MakeColumnVector(point_out_fragment, &point_out_vec);
      
      index_t new_point = -1;
      double new_dist = DBL_MAX;
      
      FindNeighbor_(tree_, point_out_vec, point_out_fragment, &new_point, &new_dist);
      candidate_neighbors_[point_out_fragment] = new_point;
      
      heap_.Put(new_dist, point_out_fragment);
      
      Vector point_in_vec;
      data_points_.MakeColumnVector(point_in_fragment, &point_in_vec);
      
      new_point = -1;
      new_dist = DBL_MAX;
      
      FindNeighbor_(tree_, point_in_vec, point_in_fragment, &new_point, &new_dist);
      candidate_neighbors_[point_in_fragment] = new_point;
      
      heap_.Put(new_dist, point_in_fragment);
      
    } // is the top priority real?
    
  } // while tree is not finished
  
  
  fx_timer_stop(mod_, "MST_computation");

  EmitResults_(results);
  
} // ComputeMST