/*
 *  n_point_multi.cc
 *  
 *
 *  Created by William March on 4/14/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_multi.h"


bool SymmetryCorrect_(ArrayList<NPointNode*>& nodes) {
  
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

index_t CheckBaseCase_(ArrayList<NPointNode*>& nodes) {
  
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
index_t FindInd_(index_t i, index_t j) {
  
  DEBUG_ASSERT(i < j);
  
  return ((i - 1) + ((j-1)*(j-2)/2));
  
} // FindInd_()

void BaseCase_(ArrayList<NPointNode*>& nodes, ResultsTensor& tuple_status,
               ResultsTensor& results) {
  
  // iterate over status tensor, skip things that are excluded
  
  for (int i = 0; i < tensor_size; i++) {
    
    if (! exclude) {
     
      matcher_.TestPointSets();
      
    }
    
  }
  
} // BaseCase_()


void DepthFirstRecursion_(ArrayList<NPointNode*>& nodes, 
                          StatusTensor& status,
                          ResultsTensor& results) {

  // check symmetry
  if (!SymmetryCorrect_(nodes)) {
    return;
  }
  index_t split_ind = CheckBaseCase_(nodes);
  
  if (split_ind < 0) {
    BaseCase_(nodes, status, results);
  }
  else {
  
    for (int perm_ind = 0; perm_ind < matcher_.num_permutations(); perm_ind++) {
    
        
      ArrayList<DRange> ranges;
      ranges.Init(n_point_impl::NChooseR(tuple_size_, 2));
      
      for (index_t i = 0; i < tuple_size_; i++) {
        
        NPointNode* node_i = nodes[i];
        
        for (index_t j = i+1; j < tuple_size_; j++) {
          
          NPointNode* node_j = nodes[j];
          
          DRange& range_ij = ranges[FindInd_(i, j)];
          matcher_.TestNodes_(node_i->bound(), node_j->bound(), i, j, perm_ind,
                              range_ij);
          
        } // for j
        
      } // for i
      
      // now, we have the ranges for this permutation, fill in the results status
      
      status.FillResults(ranges);
      
    } // for permutations
    
    // TODO: need to be able to tell if something subsumes, and perform it here
    
    // now, split and recurse
    
    NPointNode* split_node = nodes[split_ind];
    
    ResultsTensor right_status;
    right_status.Copy(status);
    
    nodes[split_ind] = split_node->left();
    
    DepthFirstRecursion_(nodes, status, results);
    
    nodes[split_ind] = split_node->right();
    DepthFirstRecursion_(nodes, right_status, results);
    
    nodes[split_ind] = split_node;

  } // not a base case
    
} // DepthFirstRecursion




