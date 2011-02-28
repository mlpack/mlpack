/*
 *  perm_free_alg.cc
 *  
 *
 *  Created by William March on 2/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "perm_free_alg.h"


void npt::PermFreeAlg::BaseCaseHelper_();

void npt::PermFreeAlg::BaseCase_(NodeTuple& nodes);

void npt::PermFreeAlg::DepthFirstRecursion_(NodeTuple& nodes) {
  
  if (CanPrune_(nodes)) {
    
    num_prunes_++;
    
  }
  else if (nodes.all_leaves()) {
    
    BaseCase_(nodes);
    
  } 
  else {
    
    
    
  } 
  
  
} // DepthFirstRecursion
