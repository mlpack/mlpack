/*
 *  n_point_testing.cc
 *  
 *
 *  Created by William March on 2/16/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point_testing.h"


int main(int argc, char* argv[]) {
  
  fx_init(argc, argv, NULL);
  
  NPointTester test;
  
  test.TestCounting();
  
  fx_done(NULL);
  
  return 0;
  
} 
