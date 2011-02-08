/*
 *  n_point_main.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

// take this out later, just debugging with it
#include "single_matcher.h"

using namespace npt;

int main(int argc, char* argv[]) {

  // read in data and parameters
  
  
  // run algorithm
  
  
  // output results
  
  
  // testing stuff
  
  //Permutations perms(3);
  //perms.Print();
  
  arma::mat lower_bds;
  arma::mat upper_bds;
  
  lower_bds << 0.0 << 0.0 << 0.0 << arma::endr
  << 0.0 << 0.0 << 0.0 << arma::endr
  << 0.0 << 0.0 << 0.0 << arma::endr;
  
  upper_bds.load("test_upper_bds.csv");
  
  SingleMatcher matcher(3, lower_bds, upper_bds);
  
  std::vector<bool> perm_ok1(6, true);
  std::cout << "Testing points: " << matcher.TestPointPair(0.5, 0, 1, perm_ok1)
  << "\n";

  //std::vector<bool> perm_ok2(6, true);
  //std::cout << "Testing boxes: " << TestHrectPair(5.0, );
  
  return 0;
  
} // main()