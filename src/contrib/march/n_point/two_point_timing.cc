/*
 *  two_point_timing.cc
 *  
 *
 *  Created by William March on 10/31/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#include <mlpack/core.h>

#include "generate_random_problem.h"
#include "generic_npt_alg.h"
#include "single_matcher.h"


PARAM_INT_REQ("num_points", "The size of the computation to run", NULL);
PARAM_INT("tuple_size", "The order of the correlation to time", NULL, 2);

using namespace mlpack;
using namespace npt;

int main(int argc, char* argv[]) {
  
  //fx_init(argc, argv, NULL);
  CLI::ParseCommandLine(argc, argv);

  GenerateRandomProblem problem_gen;
  
  int num_points = CLI::GetParam<int>("num_points");
  
  int tuple_size = CLI::GetParam<int>("tuple_size");
  
  arma::mat data_mat(3, num_points);
  problem_gen.GenerateRandomSet(data_mat);
  
  arma::colvec weights(num_points);

  arma::mat matcher_lower_bounds(tuple_size, tuple_size);
  arma::mat matcher_upper_bounds(tuple_size, tuple_size);
  
  problem_gen.GenerateRandomMatcher(matcher_lower_bounds, matcher_upper_bounds);
  
  CLI::StartTimer("single_bandwidth_time");
  
  std::vector<size_t> old_from_new_data;
  
  NptNode* data_tree = new NptNode(data_mat, old_from_new_data);

  
  std::vector<arma::mat*> comp_mats(tuple_size);
  std::vector<int> comp_multi(2);
  comp_multi[0] = 0;
  comp_multi[1] = 0;
  std::vector<arma::colvec*> comp_weights(tuple_size);
  std::vector<NptNode*> comp_trees(tuple_size);
  std::vector<std::vector<size_t>*> old_from_new_list(tuple_size);
  
  for (int i = 0; i < tuple_size; i++) {
    comp_mats[i] = &data_mat;
    comp_weights[i] = &weights;
    comp_multi[1]++;
    comp_trees[i] = data_tree;
    old_from_new_list[i] = &old_from_new_data;
  }
  
  SingleMatcher matcher(comp_mats, comp_weights, old_from_new_list,
                        matcher_lower_bounds,
                        matcher_upper_bounds);
  
  GenericNptAlg<SingleMatcher> alg(comp_trees, 
                                   //comp_multi, 
                                   matcher);
  
  alg.Compute();
  
  
  CLI::StopTimer("single_bandwidth_time");
  
  Log::Info << std::endl << "Single bandwidth num tuples: " <<  matcher.results();
  
  Log::Info << std::endl << std::endl;
  
  return 0;
  
} // main