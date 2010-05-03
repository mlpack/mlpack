/*
 *  n_point_nodes.cc
 *  
 *
 *  Created by William March on 4/28/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_nodes.h"

// fills inds with the indices in the range list that need to be recomputed
// TODO: test me!
void NodeTuple::FindInvalidIndices_(ArrayList<index_t>* inds) {
  
  index_t bad_ind = ind_to_split_ - 1;
  inds->PushBackCopy(bad_ind);
  
  for (index_t i = 1; i < ind_to_split_; i++) {
    
    bad_ind += tuple_size_ - 1 - i;
    inds->PushBackCopy(bad_ind);
    
  } // i < k
  
  bad_ind += tuple_size_ - ind_to_split_;
  for (index_t i = ind_to_split_ + 1; i < tuple_size_; i++) {

    bad_ind++;
    inds->PushBackCopy(bad_ind);
  } // i > k
  
} // FindInvalidIndices_()

void NodeTuple::UpdateIndices_(index_t split_ind, 
                               ArrayList<index_t>& invalid_indices) {
  
  NPointNode* new_node = node_list_[split_ind];
  
  int split_size = -1;
  
  index_t node_ind = 0;
  for (index_t i = 0; i < tuple_size_ - 1; i++) {
    
    NPointNode* node_i = node_list_[node_ind];
    
    if ((node_i->count() > split_size) && !node_i->is_leaf()) {
      ind_to_split_ = node_ind;
      split_size = node_i->count();
    }
    
    // don't compute the self-distance
    if (i == split_ind) {
      node_ind++;
    }
    
    
    double min_dist_sq = node_i->bound().MinDistanceSq(new_node->bound());
    double max_dist_sq = node_i->bound().MaxDistanceSq(new_node->bound());
    
    DRange this_range;
    this_range.Init(min_dist_sq, max_dist_sq);
    
    index_t bad_ind = invalid_indices[i];
    
    ranges_[bad_ind] = this_range;
    
    index_t upper_bad_ind = input_to_upper_[bad_ind];
    index_t lower_bad_ind = input_to_lower_[bad_ind];
    
    sorted_upper_[upper_bad_ind] = std::pair<double, index_t>(max_dist_sq,
                                                              upper_bad_ind);
    sorted_lower_[lower_bad_ind] = std::pair<double, index_t>(min_dist_sq,
                                                              lower_bad_ind);
    
    node_ind++;
    
  } // for i
  
  // need to resort them
  // TODO: see if there is a way to take advantage of them being somewhat 
  // sorted already
  std::sort(sorted_upper_.begin(), sorted_upper_.end());
  std::sort(sorted_lower_.begin(), sorted_lower_.end());
  
  for (index_t i = 0; i < sorted_upper_.size(); i++) {
    
    input_to_upper_[sorted_upper_[i].second] = i;
    input_to_lower_[sorted_lower_[i].second] = i;
    
  } // fill in input to sorted arrays
  
  if (split_size < 0) {
    ind_to_split_ = -1;
  }
  
} // UpdateIndices

// fills in the node like Init, but doesn't have to recompute all pairwise 
// distances
void NodeTuple::SplitHelper_(ArrayList<NPointNode*>& node_list, 
                             index_t split_ind,
                             ArrayList<DRange>& ranges_in,
                             ArrayList<std::pair<double, index_t> >& upper_in,
                             ArrayList<std::pair<double, index_t> >& lower_in,
                             ArrayList<index_t>& invalid_indices,
                             ArrayList<index_t>& in_to_hi,
                             ArrayList<index_t>& in_to_lo) {

  node_list_.Swap(&node_list);
  sorted_upper_.InitCopy(upper_in);
  sorted_lower_.InitCopy(lower_in);
  ranges_.InitCopy(ranges_in);
  input_to_upper_.InitCopy(in_to_hi);
  input_to_lower_.InitCopy(in_to_lo);
  
  tuple_size_ = invalid_indices.size() + 1;
  
  UpdateIndices_(split_ind, invalid_indices);
    
  
} // SplitHelper_()



// this node will be the left, the other will be the right
// TODO: Add symmetry checking here, before messing with 
bool NodeTuple::PerformSplit(NodeTuple* right_node) {
  
  NPointNode* split_node = node_list_[ind_to_split_];
  DEBUG_ASSERT(!(split_node->is_leaf()));
  
  // contains all indices in the list of ranges that need to be computed
  ArrayList<index_t> invalid_indices;
  invalid_indices.Init();
  FindInvalidIndices_(&invalid_indices);
  DEBUG_ASSERT(invalid_indices.size() == tuple_size_ - 1);
  
  // form the right node here, give it the three arrays and the list of 
  // invalid indices
  
  ArrayList<NPointNode*> right_node_list;
  right_node_list.InitCopy(node_list_);
  right_node_list[ind_to_split_] = split_node->right();
  // TODO: add symmetry check here, return NULL if bad
  right_node->SplitHelper_(right_node_list, ind_to_split_, ranges_, 
                           sorted_upper_, sorted_lower_, 
                           invalid_indices, input_to_upper_, input_to_lower_);
  
  node_list_[ind_to_split_] = split_node->left();
  // TODO: add symmetry check here, return false if bad
  UpdateIndices_(ind_to_split_, invalid_indices);
  
  return true;
  
  
} // PerformSplit
