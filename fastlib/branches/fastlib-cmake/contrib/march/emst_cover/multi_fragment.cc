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
        //double this_dist = sqrt(la::DistanceSqEuclidean(point, point_i));
        
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
    
    DEBUG_ASSERT(point_index >= tree->right()->begin());
    UpdateMemberships_(tree->right(), point_index);
    
    if (tree->left()->stat().component_membership() 
        == tree->right()->stat().component_membership()) {
      tree->stat().set_component_membership(tree->left()->stat().component_membership());
    }
    
  }
  
} // UpdateMemberships_()

void MultiFragment::MergeQueues_(index_t comp1, index_t comp2) {
  
  index_t union_comp = connections_.Find(comp1);
  
  MinHeap<double, CandidateEdge> *queue_rec, *queue_give;
  index_t ind_rec, ind_give;
  
  if (comp1 == union_comp) {
    
    // merge comp2 into comp 1
    
    queue_rec = &(fragment_queues_[comp1]);
    queue_give = &(fragment_queues_[comp2]);
    
    ind_rec = comp1;
    ind_give = comp2;

  }
  else {
    
    DEBUG_ASSERT(comp2 == union_comp);

    queue_rec = &(fragment_queues_[comp2]);
    queue_give = &(fragment_queues_[comp1]);

    ind_rec = comp2;
    ind_give = comp1;

  }
  
  //DEBUG_ASSERT(queue_give->size() == fragment_sizes_[ind_give]);
  
  
  //for (index_t i = 0; i < queue_give->size(); i++) {
  while (queue_give->size() > 0) {
    
    double key = queue_give->top_key();
    CandidateEdge val = queue_give->Pop();

    //if (val.is_valid(connections_)) {
    // want to make sure that each point has a candidate
      queue_rec->Put(key, val);
    //}
  } // for i

  //printf("%d, %d\n", fragment_sizes_[ind_rec], fragment_sizes_[ind_give]);
  index_t new_size = fragment_sizes_[ind_rec] 
  + fragment_sizes_[ind_give];
  fragment_sizes_[ind_rec] = -1;
  fragment_sizes_[ind_give] = -1;
  fragment_sizes_[union_comp] = new_size;
  fragment_size_heap_.Put(new_size, union_comp);
  
} // MergeQueues_

void MultiFragment::ComputeMST(Matrix* results) {

  fx_timer_start(mod_, "MST_computation");
  
  while (number_of_edges_ < number_of_points_ - 1) {
    
    // find smallest fragment
    
    /*
    index_t current_frag;
    index_t current_size = -1;
    while (current_size < 0) { // look for a real component index
      current_frag = fragment_size_heap_.Pop();
      current_size = fragment_sizes_[current_frag];
    }
     */
    
    // not quite right either, the component I just popped may not be the one 
    // that still exists
    
    
    bool real_frag = 0;
    index_t current_frag;
    while (!real_frag) {
      index_t frag_size = fragment_size_heap_.top_key();
      current_frag = fragment_size_heap_.Pop();
      real_frag = ((current_frag == connections_.Find(current_frag))
                   && (frag_size == fragment_sizes_[current_frag]));
      //real_frag = (current_frag == connections_.Find(current_frag));
    } 
     
    
    //DEBUG_ASSERT(fragment_size_heap_.size() == number_of_points_ - number_of_edges_ - 1);
     
    // this should make it act like single-tree, since it will always grow 
    // one component
    //index_t current_frag = connections_.Find(0);
    
    // there should be one candidate point per point in the fragment
    // not true, when merging can wind up with more than one
    //DEBUG_ASSERT(fragment_queues_[current_frag].size() == fragment_sizes_[current_frag]);
    
    while (1) {
      
      double dist = fragment_queues_[current_frag].top_key();
      CandidateEdge this_edge = fragment_queues_[current_frag].Pop();
      
      DEBUG_ASSERT(current_frag == connections_.Find(this_edge.point_in()));
      
      index_t point_in = this_edge.point_in();
      index_t point_out = this_edge.point_out();
      
      /*
      index_t point_in = fragment_queues_[current_frag].Pop();
      index_t point_out = candidate_neighbors_[point_in];
      */
      
#ifdef DEBUG
      
      Vector in_vec, out_vec;
      data_points_.MakeColumnVector(point_in, &in_vec);
      data_points_.MakeColumnVector(point_out, &out_vec);

      //double true_dist = sqrt(la::DistanceSqEuclidean(in_vec, out_vec));
      double true_dist = la::DistanceSqEuclidean(in_vec, out_vec);
      
      DEBUG_ASSERT((true_dist - dist) < 10e-5);
      
#endif
      
      index_t comp_in = connections_.Find(point_in);
      index_t comp_out = connections_.Find(point_out);
      DEBUG_ASSERT(comp_in == current_frag);
      
      if (comp_in == comp_out) {
      //if (!(this_edge.is_valid(connections_))) {  
        // the link isn't real, find a real one and re-insert
        
        Vector point_vec;
        data_points_.MakeColumnVector(point_in, &point_vec);
        
        index_t new_point = -1;
        double new_dist = DBL_MAX;
        
        FindNeighbor_(tree_, point_vec, point_in, &new_point, 
                      &new_dist);
        //candidate_neighbors_[point_in] = new_point;
        
        //fragment_queues_[current_frag].Put(new_dist, point_in);
        CandidateEdge new_edge;
        new_edge.Init(point_in, new_point);
        fragment_queues_[current_frag].Put(new_dist, new_edge);
        
      } // link not real
      else {
        // the link is real, add the edge + update the queues, break
        
        // minus 1 because one has been popped
        DEBUG_ASSERT(fragment_sizes_[comp_in] - 1 == fragment_queues_[comp_in].size());
        DEBUG_ASSERT(fragment_sizes_[comp_out] == fragment_queues_[comp_out].size());

        AddEdge_(point_in, point_out, dist);
        connections_.Union(point_in, point_out);
        
        // merge the fragment_queues & fragment sizes
        MergeQueues_(comp_in, comp_out);
        
        
        index_t new_comp = connections_.Find(point_in);
        DEBUG_ASSERT(new_comp == comp_in || new_comp == comp_out);

        DEBUG_ASSERT(fragment_sizes_[new_comp] - 1 == fragment_queues_[new_comp].size());
        
        UpdateMemberships_(tree_, point_in);
        UpdateMemberships_(tree_, point_out);
        // update the other point again?
        
        Vector point_out_vec;
        data_points_.MakeColumnVector(point_out, &point_out_vec);
        
        index_t new_point = -1;
        double new_dist = DBL_MAX;
        
        // don't need this, it should already have a neighbor in its own queue
        /*
        FindNeighbor_(tree_, point_out_vec, point_out, &new_point, 
                      &new_dist);
        //candidate_neighbors_[point_out] = new_point;
        //fragment_queues_[new_comp].Put(new_dist, point_out);
        
        CandidateEdge new_out_edge;
        new_out_edge.Init(point_out, new_point);
        fragment_queues_[new_comp].Put(new_dist, new_out_edge);
        */
        
        Vector point_in_vec;
        data_points_.MakeColumnVector(point_in, &point_in_vec);
        
        new_point = -1;
        new_dist = DBL_MAX;
        
        FindNeighbor_(tree_, point_in_vec, point_in, &new_point, 
                      &new_dist);
        //candidate_neighbors_[point_in] = new_point;
        //fragment_queues_[new_comp].Put(new_dist, point_in);
        CandidateEdge new_in_edge;
        new_in_edge.Init(point_in, new_point);
        fragment_queues_[new_comp].Put(new_dist, new_in_edge);
        
        DEBUG_ASSERT(fragment_queues_[new_comp].size() == fragment_sizes_[new_comp]);
        
        break; // time for a new smallest component
        
      } // link real
      
      
    } // smallest fragment doesn't have real priority
    
    
    
  } // while more than one fragment
  
  fx_timer_stop(mod_, "MST_computation");

  EmitResults_(results);
  
} // ComputeMST()


