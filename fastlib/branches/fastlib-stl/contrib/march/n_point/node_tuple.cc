/*
 *  node_tuple.cc
 *  
 *
 *  Created by William March on 2/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "node_tuple.h"


bool npt::NodeTuple::CheckSymmetry(const std::vector<NptNode*>& nodes, 
                                   index_t split_ind, bool is_left) {
  
  // only check the new node for symmetry with respect to the others
  if (is_left) {
    for (index_t i = 0; i < split_ind; i++) {
      
      if (nodes[split_ind]->left()->end() <= nodes[i]->begin()) {
        return false;
      }
      
    } // for i
    
    for (index_t i = split_ind + 1; i < tuple_size_; i++) {
      
      if (nodes[i]->end() <= nodes[split_ind]->left()->begin()) {
        return false;
      } 
      
    } // for i
  }
  else {
    for (index_t i = 0; i < split_ind; i++) {
      
      if (nodes[split_ind]->right()->end() <= nodes[i]->begin()) {
        return false;
      }
      
    } // for i
    
    for (index_t i = split_ind + 1; i < tuple_size_; i++) {
      
      if (nodes[i]->end() <= nodes[split_ind]->right()->begin()) {
        return false;
      } 
      
    } // for i    
  }
  return true;
  
} // CheckSymmetry_



void npt::NodeTuple::FindInvalidIndices_(std::vector<index_t>& inds) {

  //std::vector<index_t> inds;
  
  // simple case
  if (tuple_size_ == 2) {
    inds.push_back(0);
  }
  else {
    
    index_t bad_ind = ind_to_split_ - 1;
    index_t bad_ind2 = 0;
    
    // the values are in a lower triangular matrix, stored in a straight array
    // this loop goes horizontally along the row ind_to_split_ until
    // we reach the diagonal
    for (index_t i = 0; i < ind_to_split_; i++) {
      
      inds.push_back(bad_ind);
      bad_ind += tuple_size_ - 1 - (i+1);
      bad_ind2 += tuple_size_ - i - 1;
      
    } // horizontal
    
    // after the diagonal, we need to go down the column
    for (index_t i = ind_to_split_+1; i < tuple_size_; i++) {
      
      inds.push_back(bad_ind2);
      bad_ind2++;
      
    }
    
  }    
    
} // FindInvalidIndices


void npt::NodeTuple::UpdateIndices_(index_t split_ind) {
  
  NptNode* new_node = node_list_[split_ind];
  
  // everything at and past new_positions[i] has been pushed back one space
  // I'll need to consider the ordering to know how things have been pushed back
  std::vector<index_t> new_positions_max;
  std::vector<index_t> new_positions_min;
  
  ind_to_split_ = -1;
  int split_count = 0;
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    // check for all leaves here
    NptNode* node_i = node_list_[i];
    
    if (!node_i->is_leaf()) {
      all_leaves_ = false;
      if (node_i->count() > split_count) {
        split_count = node_i->count();
        ind_to_split_ = i;
      }
    }
    
    if (i != split_ind) {

      
      double max_dist_sq = new_node->bound().MaxDistanceSq(node_i->bound());
      double min_dist_sq = new_node->bound().MinDistanceSq(node_i->bound());
    
      for (index_t max_ind = 0; max_ind < sorted_upper_.size(); max_ind++) {
        
        if (max_dist_sq <= sorted_upper_[max_ind]) {
          
          sorted_upper_.insert(max_ind, max_dist_sq);
          
          // now, keep track of how the other indices changed
          new_positions_max.push_back(max_ind);
          
          break;
          
        }
        
      } // max_ind
      
      for (index_t min_ind = 0; min_ind < sorted_upper_.size(); min_ind++) {
        
        if (min_dist_sq <= sorted_lower_[min_ind]) {
          
          sorted_lower_.insert(min_ind, min_dist_sq);
          
          // now, keep track of how the other indices changed
          new_positions_min.push_back(min_ind);
          
          break;
          
        }
        
      } // min_ind
      
      
    } // not equal to split ind
    
  } // for i
  
  // now handle the indices
  for (index_t i = 0; i < tuple_size_; i++) {
    
    if (i == split_ind) { 
      continue;
    }
    
    // erase entry map_upper_to_sorted_(i, split_ind) + shifts from
    // sorted_upper_
    
    
  } // for i
  
  
} // UpdateIndices()




