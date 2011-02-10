/*
 *  n_point_main.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "single_bandwidth_alg.h"
#include "naive_alg.h"

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
  
  arma::mat data;
  data.load("test_data.csv");
  
  arma::colvec weights(data.n_cols);
  weights.fill(1.0);
  
  index_t leaf_size = 1;
  
  SingleBandwidthAlg single_alg(data, weights, leaf_size, 
                                lower_bds, upper_bds);
  
  single_alg.ComputeCounts();
  
  std::cout << "Single Bandwidth num tuples: " << single_alg.num_tuples() << "\n";
  
  NaiveAlg naive_alg(data, weights, lower_bds, upper_bds);

  naive_alg.ComputeCounts();
  
  std::cout << "Naive num tuples: " << naive_alg.num_tuples() << "\n";
  
  /*
  SingleMatcher matcher(3, lower_bds, upper_bds);
  
  std::vector<bool> perm_ok1(6, true);
  std::cout << "Testing points: " << matcher.TestPointPair(0.5, 0, 1, perm_ok1)
  << "\n";
  */
  
  //std::vector<bool> perm_ok2(6, true);
  //std::cout << "Testing boxes: " << TestHrectPair(5.0, );
  
  return 0;
  
} // main()