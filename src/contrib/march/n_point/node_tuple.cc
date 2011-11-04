/*
 *  node_tuple.cc
 *  
 *
 *  Created by William March on 2/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "node_tuple.h"



// Invariant: a tuple of points must go in increasing order of index (within
// the data and random parts of the tuple)
// Therefore, for a tuple of nodes to be valid, it must be possible that some 
// tuple of points taken from it satisfies this requirement
// Therefore, if the begin of one node is larger than the end of a node that 
// preceeds it, the symmetry is violated

// NOTE: this assumes that nodes from the same data set are in consecutive 
// order in the node_list
bool npt::NodeTuple::CheckSymmetry(size_t split_ind, bool is_left) {
  
  
  int start_point = -1;
  int end_point = tuple_size_;
  
  // This uses the assumption that the symmetry was correct before
  // Therefore, we only need to check the new node against the others from the
  // same set
  int this_id = same_nodes_[split_ind];
  // start_point is the first index i where same_nodes_[i] = same_nodes_[split_ind]
  // end_point is the last one where this is true
  
  for (unsigned int i = 0; i <= split_ind; i++) {
    if (this_id == same_nodes_[i]) {
      start_point = i;
      break;
    }
  }
  // If the true end_point is tuple_size, it won't get set here, but it was
  // already set above.
  for (int i = split_ind + 1; i < tuple_size_; i++) {
    if (this_id != same_nodes_[i]) {
      end_point = i;
      break;
    }
  }
  
  assert(start_point >= 0);
  
  // only check the new node for symmetry with respect to the others
  if (is_left) {
    for (unsigned int i = start_point; i < split_ind; i++) {
      
      if (node_list_[split_ind]->left()->end() <= node_list_[i]->begin()) {
        return false;
      }
      
    } // for i
    
    for (int i = split_ind; i < end_point; i++) {
      
      if (node_list_[i]->end() <= node_list_[split_ind]->left()->begin()) {
        return false;
      } 
      
    } // for i
    
  }
  else { // is right

    for (unsigned int i = start_point; i < split_ind; i++) {
      
      if (node_list_[split_ind]->right()->end() <= node_list_[i]->begin()) {
        return false;
      }
      
    } // for i
    
    for (int i = split_ind + 1; i < end_point; i++) {
      
      if (node_list_[i]->end() <= node_list_[split_ind]->right()->begin()) {
        return false;
      } 
      
    } // for i    
    
  }
  return true;
  
} // CheckSymmetry_



void npt::NodeTuple::UpdateSplitInd_() {
  
  unsigned int split_size = 0;
  
  all_leaves_ = true;
  
  for (int i = 0; i < tuple_size_; i++) {
    
    if (!(node_list_[i]->is_leaf())) {
      
      all_leaves_ = false;

      if (node_list_[i]->count() > split_size) {
        split_size = node_list_[i]->count();
        ind_to_split_ = i;
      }
        
    } // is a leaf?
    
  } // for i
  
} // UpdateSplitInd_


