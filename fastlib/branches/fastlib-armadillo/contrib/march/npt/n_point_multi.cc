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


void DepthFirstRecursion_(ArrayList<NPointNode*>& nodes, 
                          ResultsTensor& tuple_status) {

  // check symmetry
  if (!SymmetryCorrect_(nodes)) {
    return;
  }
  
  ArrayList<ResultsTensor> permutation_ok;
  permutation_ok.Init(num_permutations);
  
  for (int i = 0; i < num_permutations; i++) {
    
    permutation_ok[i].Copy(tuple_status);
    
  }
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    NPointNode* node_i = nodes[i];
    
    for (index_t j = i+1; j < tuple_size_; j++) {
      
      NPointNode* node_j = nodes[j];
    
      // fill in perm_status based on info from i and j
      
      matcher_.TestNodes_(node_i->bound(), node_j->bound(),
                          permutation_ok);
      
    } // node k
    
  } // for node j
  
      
  // update tuple_status as appropriate
  
  // perform subsuming prunes here
  
  // choose a split node, pass to children
  
} // DepthFirstRecursion