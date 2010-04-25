/*
 *  n_point_impl.h
 *  
 *
 *  Created by William March on 4/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_IMPL_H
#define N_POINT_IMPL_H

#include "fastlib/fastlib.h"

// Combinatorial functions I'll need in more than one place
namespace n_point_impl {
 
  int NChooseR(int n, int r) {
    
    DEBUG_ASSERT(n >= 0);
    DEBUG_ASSERT(r >= 0);
    
    if(n < r) return 0;
    
    int divisor = 1;
    int multiplier = n;
    
    int answer = 1;
    
    while (divisor <= r) {
      
      answer = (answer * multiplier) / divisor;
      
      multiplier--;
      divisor++;
      
    } 
    
    return answer;
    
  } // NChooseR
  
  
  
} // namespace



#endif