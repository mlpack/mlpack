/*
 *  multi_fragment.cc
 *  
 *
 *  Created by William March on 1/19/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "multi_fragment.h"

void MultiFragment::FindNeighbor_(MFTree* tree, const Vector& point, 
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
    
    if (tree->left()->stat().component_membership() != point_index) {
      left_dist = tree->left()->bound().MinDistanceSq(point);
    }
    else {
      left_dist = DBL_MAX;
    }
    
    if (tree->right()->stat().component_membership() != point_index) {
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

void MultiFragment::UpdateMemberships_(MFTree* tree, index_t point_index) {
  
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

void MultiFragment::MergeQueues_(index_t comp1, index_t comp2) {
  
  index_t union_comp = connections_.Find(comp1);
  
  MinHeap<double, index_t> queue_rec, queue_give;
  index_t ind_rec, ind_give;
  
  if (comp1 == union_comp) {
    
    // merge comp2 into comp 1
    
    queue_rec = fragment_queues_[comp1];
    queue_give = fragment_queues_[comp2];
    
    ind_rec = comp1;
    ind_give = comp2;

  }
  else {
    
    DEBUG_ASSERT(comp2 == union_comp);

    queue_rec = fragment_queues_[comp2];
    queue_give = fragment_queues_[comp1];

    ind_rec = comp2;
    ind_give = comp1;

  }
  
  index_t new_size = fragment_sizes_[ind_rec] 
                     + fragment_sizes_[ind_give];
  fragment_sizes_[ind_rec] = new_size;
  fragment_sizes_[ind_give] = -1;
  fragment_size_heap_.Put(new_size, ind_rec);
  
  
  for (index_t i = 0; i < queue_give.size(); i++) {
    
    double key = queue_give.top_key();
    index_t val = queue_give.Pop();
    queue_rec.Put(key, val);
    
  } // for i
  
} // MergeQueues_

void MultiFragment::ComputeMST(Matrix* results) {

  while (number_of_edges_ < number_of_points_ - 1) {
    
    // find smallest fragment
    
    index_t current_frag;
    index_t current_size = -1;
    while (current_size < 0) { // look for a real component index
      current_frag = fragment_size_heap_.Pop();
      current_size = fragment_sizes_[current_frag];
    }
      
    while (1) {
      
      double dist = fragment_queues_[current_frag].top_key();
      index_t point_in_frag = fragment_queues_[current_frag].Pop();
      index_t point_out_frag = candidate_neighbors_[point_in_frag];
      
      index_t point_in_comp = connections_.Find(point_in_frag);
      index_t point_out_comp = connections_.Find(point_out_frag);
      DEBUG_ASSERT(point_in_comp == current_frag);
      
      if (point_in_comp == point_out_comp) {
        // the link isn't real, find a real one and re-insert
        
        Vector point_vec;
        data_points_.MakeColumnVector(point_in_frag, &point_vec);
        
        index_t new_point = -1;
        double new_dist = DBL_MAX;
        
        FindNeighbor_(tree_, point_vec, point_in_frag, &new_point, 
                      &new_dist);
        candidate_neighbors_[point_in_frag] = new_point;
        
        fragment_queues_[current_frag].Put(new_dist, point_in_frag);
        
        
      } // link not real
      else {
        // the link is real, add the edge + update the queues, break

        AddEdge_(point_in_frag, point_out_frag, dist);
        connections_.Union(point_in_comp, point_out_comp);
        
        // merge the fragment_queues & fragment sizes
        MergeQueues_(point_in_comp, point_out_comp);
        
        index_t new_comp = connections_.Find(point_in_frag);
        
        UpdateMemberships_(tree_, point_in_frag);
        UpdateMemberships_(tree_, point_out_frag);
        
        Vector point_out_vec;
        data_points_.MakeColumnVector(point_out_frag, &point_out_vec);
        
        index_t new_point = -1;
        double new_dist = DBL_MAX;
        
        FindNeighbor_(tree_, point_out_vec, point_out_frag, &new_point, 
                      &new_dist);
        candidate_neighbors_[point_out_frag] = new_point;
        
        fragment_queues_[new_comp].Put(new_dist, point_out_frag);
        
        Vector point_in_vec;
        data_points_.MakeColumnVector(point_in_frag, &point_in_vec);
        
        new_point = -1;
        new_dist = DBL_MAX;
        
        FindNeighbor_(tree_, point_in_vec, point_in_frag, &new_point, 
                      &new_dist);
        candidate_neighbors_[point_in_frag] = new_point;
        
        fragment_queues_[new_comp].Put(new_dist, point_in_frag);
        
        break; // time for a new smallest component
        
      } // link real
      
      
    } // smallest fragment doesn't have real priority
    
    
    
  } // while more than one fragment
  
  
  EmitResults_(results);
  
} // ComputeMST()


