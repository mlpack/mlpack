/*
 *  n_point_impl.cc
 *  
 *
 *  Created by William March on 4/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_impl.h"

int n_point_impl::NChooseR(int n, int r) {
  
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