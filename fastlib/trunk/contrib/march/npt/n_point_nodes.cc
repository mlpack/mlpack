/*
 *  n_point_nodes.cc
 *  
 *
 *  Created by William March on 4/28/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_nodes.h"




void NodeTuple::UpdateIndices_(index_t split_ind, 
                               ArrayList<index_t>& invalid_indices) {
  
  NPointNode* new_node = node_list_[split_ind];
  
  int split_size = -1;
  ind_to_split_ = -1;
  
  index_t invalid_ind = 0;
  
  all_leaves_ = true;
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    NPointNode* node_i = node_list_[i];

    if (!(node_i->is_leaf())) {
      all_leaves_ = false;
    
      if (node_i->count() > split_size) {
        ind_to_split_ = i;
        split_size = node_i->count();
      }

    } // not a leaf
    
    // don't compute the self-distance
    if (i != split_ind) {
    
      double min_dist_sq = node_i->bound().MinDistanceSq(new_node->bound());
      double max_dist_sq = node_i->bound().MaxDistanceSq(new_node->bound());
      
      DRange this_range;
      this_range.Init(min_dist_sq, max_dist_sq);
      
      index_t bad_ind = invalid_indices[invalid_ind];
      invalid_ind++;
      
      ranges_[bad_ind] = this_range;
      
      // Is this right?  Am I using these correctly
      index_t upper_bad_ind = input_to_upper_[bad_ind];
      index_t lower_bad_ind = input_to_lower_[bad_ind];
      
      sorted_upper_[upper_bad_ind] = std::pair<double, index_t>(max_dist_sq,
                                                                bad_ind);
      sorted_lower_[lower_bad_ind] = std::pair<double, index_t>(min_dist_sq,
                                                                bad_ind);

    }
    
  } // for i
  
  /*
  printf("\nall_leaves: %d, ind_to_split_: %d\n", all_leaves_, ind_to_split_);
  printf("tuple_size: %d\n", tuple_size_);
  for (int i = 0; i < tuple_size_; i++) {
    printf("node(%d)->count(): %d\n", i, node_list_[i]->count());
  }
  printf("\n");
  */
  
  // need to resort them
  // TODO: see if there is a way to take advantage of them being somewhat 
  // sorted already
  std::sort(sorted_upper_.begin(), sorted_upper_.end());
  std::sort(sorted_lower_.begin(), sorted_lower_.end());
  
  for (index_t i = 0; i < sorted_upper_.size(); i++) {
    
    input_to_upper_[sorted_upper_[i].second] = i;
    input_to_lower_[sorted_lower_[i].second] = i;
    
  } // fill in input to sorted arrays
  
  /*
  if (split_size < 0) {
    ind_to_split_ = -1;
  }
   */
  
  DEBUG_ASSERT(node_list_.size() > 0);
  
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

  node_list_.InitCopy(node_list);
  sorted_upper_.InitCopy(upper_in);
  sorted_lower_.InitCopy(lower_in);
  ranges_.InitCopy(ranges_in);
  input_to_upper_.InitCopy(in_to_hi);
  input_to_lower_.InitCopy(in_to_lo);
  
  tuple_size_ = node_list_.size();
  
  //printf("right_node tuple_size: %d\n", tuple_size_);
  
  UpdateIndices_(split_ind, invalid_indices);
  
  DEBUG_ASSERT(node_list_.size() > 0);
    
  
} // SplitHelper_()


bool CheckSymmetry_(ArrayList<NPointNode*>& node_list) {
  
  DEBUG_ASSERT(node_list.size() > 1);
  
  for (index_t i = 0; i < node_list.size(); i++) {
    
    NPointNode* node_i = node_list[i];
    
    for (index_t j = i+1; j < node_list.size(); j++) {
      
      NPointNode* node_j = node_list[j];
      
      if (node_j->end() <= node_i->begin()) {
        return false;
      }
      
    } // for j
  } // for i
  
  return true;
  
} // CheckSymmetry_


void NodeTuple::PerformSplit(NodeTuple*& left_node, NodeTuple*& right_node,
                             ArrayList<ArrayList<index_t> >& invalid_index_list) {
  
  NPointNode* split_node = node_list_[ind_to_split_];
  DEBUG_ASSERT(!(split_node->is_leaf()));
  
  // contains all indices in the list of ranges that need to be computed
  ArrayList<index_t>& invalid_indices = invalid_index_list[ind_to_split_];
  
  // form the children here, give it the three arrays and the list of 
  // invalid indices

  /*
  ArrayList<NPointNode*> left_node_list;
  left_node_list.InitCopy(node_list_);
  left_node_list[ind_to_split_] = split_node->left();
  */
  /*
  ArrayList<NPointNode*> right_node_list;
  right_node_list.InitCopy(node_list_);
  right_node_list[ind_to_split_] = split_node->right();
  */
  
  node_list_[ind_to_split_] = split_node->right();
  
  if (CheckSymmetry_(node_list_)) {
    right_node->SplitHelper_(node_list_, ind_to_split_, ranges_, 
                             sorted_upper_, sorted_lower_, 
                             invalid_indices, input_to_upper_, input_to_lower_);
    //right_node->Init(node_list_);
    DEBUG_ASSERT(right_node->node_list(0));
  }
  else {
    right_node = NULL;
  }

  
  node_list_[ind_to_split_] = split_node->left();
  
  if (CheckSymmetry_(node_list_)) {
    left_node->SplitHelper_(node_list_, ind_to_split_, ranges_, 
                            sorted_upper_, sorted_lower_, 
                            invalid_indices, input_to_upper_, input_to_lower_);
    //DEBUG_ASSERT(left_node->node_list(0));
    //left_node->Init(node_list_);
  }
  else {
    left_node = NULL;
  }
  
  //DEBUG_ASSERT(left_node->node_list(0));
  
  
  node_list_[ind_to_split_] = split_node;
  
  /*
  node_list_[ind_to_split_] = split_node->left();
  
  //printf("left_node tuple_size: %d\n", tuple_size_);
  
  
  if (CheckSymmetry_(node_list_)) {
  
    UpdateIndices_(ind_to_split_, invalid_indices);
    
    return true;
    
  }
  else {
      
    return false;
    
  }
  */
  
} // PerformSplit
