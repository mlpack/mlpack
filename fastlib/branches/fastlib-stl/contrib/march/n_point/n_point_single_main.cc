/*
 *  n_point_main.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "fastlib/fastlib.h"
#include <fastlib/fx/io.h>

//#include "single_bandwidth_alg.h"
//#include "perm_free_alg.h"
//#include "multi_bandwidth_alg.h"
#include "generic_npt_alg.h"
#include "single_matcher.h"

PARAM_MODULE("n_point_single",
             "Parameters for generic, single-matcher n-point computation.");
PARAM_STRING_REQ("data", "Point coordinates.", NULL);
PARAM_INT_REQ("num_random", "The number of random sets that appear in the tuple.", NULL);
PARAM_FLAG("two_sets", "Are we using two different sets (i.e. data and random)", NULL);
PARAM_STRING("random", "Poisson set coordinates.", NULL, "fake");
PARAM_FLAG("weighted_computation", "Specify if computing with pointwise weights", NULL)
PARAM_STRING("weights", "Optional data weights.", NULL, "default_weights.csv");
PARAM_STRING("random_weights", "Optional weights on Poisson set.", NULL, "default_weights.csv");
//PARAM_STRING_REQ("matcher_dists", "The distances in the matcher, stored in a symmetric matrix.",
//                 NULL);
//PARAM_DOUBLE("bandwidth", "Thickness of the matcher", NULL,
//      1.0)
PARAM_STRING_REQ("matcher_lower_bounds", "The lower bound distances for the matcher.", NULL)
PARAM_STRING_REQ("matcher_upper_bounds", "The upper bound distances for the matcher.", NULL)
//PARAM_INT("leaf_size", "Max number of points in a leaf node", NULL, 1);
PARAM_FLAG("do_naive", "Permform Naive computation", NULL);
PARAM_FLAG("do_single_bandwidth", "Permform old (Moore & Gray) tree computation", NULL);
//PARAM_FLAG("do_perm_free", "Tree computation with alternative pruning rule", NULL);

using namespace mlpack;
using namespace npt;

int main(int argc, char* argv[]) {

  //fx_init(argc, argv, NULL);
  IO::ParseCommandLine(argc, argv);
  
  IO::Info << "Parsed command line.\n";
  
  // read in data and parameters
  
  //std::string data_filename = fx_param_str(NULL, "data", "test_npt_pts.csv");
  std::string data_filename = IO::GetParam<std::string>("data");
  
  arma::mat data_in, data_mat;
  data_in.load(data_filename, arma::raw_ascii);
  
  // THIS IS BAD: do it better
  if (data_in.n_rows > data_mat.n_cols) {
    data_mat = arma::trans(data_in);
  }
  else {
    data_mat = data_in;
  }
  data_in.reset();
  
  //arma::mat data_out = arma::trans(data_mat);
  //data_out.save("3pt_test_data.csv", arma::raw_ascii);
  
  arma::colvec weights;  
  //if (fx_param_exists(NULL, "weights")) {
  
  //bool has_weights = IO::HasParam("weighted_computation");
  //IO::Info << "has_weights: " << has_weights << "\n";
  
  //if (IO::HasParam("weighted_computation")) {
  if (IO::GetParam<bool>("weighted_computation")) {
    weights.load(IO::GetParam<std::string>("weights"));
  }
  else {
    weights.set_size(data_mat.n_cols);
    weights.fill(1.0);
  }
  
  
  arma::mat random_mat;
  arma::colvec random_weights;
  if (IO::GetParam<bool>("two_sets")) {
  //if (IO::HasParam("random")) {
    std::string random_filename = IO::GetParam<std::string>("random");
    
    arma::mat random_in;
    random_in.load(random_filename, arma::raw_ascii);
    
    // THIS IS BAD: do it better
    if (random_in.n_rows > random_mat.n_cols) {
      random_mat = arma::trans(random_in);
    }
    else {
      random_mat = random_in;
    }
    random_in.reset();
    
    //arma::mat data_out = arma::trans(data_mat);
    //data_out.save("3pt_test_data.csv", arma::raw_ascii);
    
    arma::colvec random_weights;  
    //if (fx_param_exists(NULL, "weights")) {
    if (IO::GetParam<bool>("weighted_computation")) {
      
  //  if (IO::HasParam("weighted_computation")) {
      random_weights.load(IO::GetParam<std::string>("random_weights"));
    }
    else {
      random_weights.set_size(random_mat.n_cols);
      random_weights.fill(1.0);
    }
    
  }
  
  //std::cout << "loaded weights\n";
  
  arma::mat matcher_lower_bounds, matcher_upper_bounds;
  
  matcher_lower_bounds.load(IO::GetParam<std::string>("matcher_lower_bounds"));
  matcher_upper_bounds.load(IO::GetParam<std::string>("matcher_upper_bounds"));
  
  //std::cout << "loaded bounds\n";
  
  int num_random = IO::GetParam<int>("num_random");
  
  int tuple_size = matcher_lower_bounds.n_cols;
  
  std::vector<arma::mat*> comp_mats(tuple_size);
  std::vector<int> comp_multi(2);
  comp_multi[0] = 0;
  comp_multi[1] = 0;
  std::vector<arma::colvec*> comp_weights(tuple_size);
  std::vector<NptNode*> comp_trees(2);
  
  for (int i = 0; i < num_random; i++) {
    comp_mats[i] = &random_mat;
    comp_weights[i] = &random_weights;
    comp_multi[0]++;
  }
  for (int i = num_random; i < tuple_size; i++) {
    comp_mats[i] = &data_mat;
    comp_weights[i] = &weights;
    comp_multi[1]++;
  }
  
  // run algorithm
  
  //if (fx_param_exists(NULL, "do_single_bandwidth")) {
  if (IO::GetParam<bool>("do_single_bandwidth")) {
  //if (IO::HasParam("do_single_bandwidth")) {
    
    //std::cout << "Doing single bandwidth.\n";
    IO::Info << "Doing single bandwidth.\n";

    //fx_timer_start(NULL, "single_bandwidth_time");
    IO::StartTimer("single_bandwidth_time");
    
    // Build the trees
    
    //IO::GetParam<int>("tree/leaf_size") = 100;
    
    std::vector<size_t> old_from_new_data;
    std::vector<size_t> old_from_new_random;
    
    
    NptNode* data_tree = new NptNode(data_mat, old_from_new_data);
    NptNode* random_tree = new NptNode(random_mat, old_from_new_random);

    std::vector<std::vector<size_t>*> old_from_new_list(tuple_size);
    
    for (int i = 0; i < num_random; i++) {
      old_from_new_list[i] = &old_from_new_random;
    }
    for (int i = num_random; i < tuple_size; i++) {
      old_from_new_list[i] = &old_from_new_data;
    }

    
    comp_trees[0] = random_tree;
    comp_trees[1] = data_tree;
    
    
    SingleMatcher matcher(comp_mats, comp_weights, old_from_new_list,
                          matcher_lower_bounds,
                          matcher_upper_bounds);
    
    GenericNptAlg<SingleMatcher> alg(comp_trees, comp_multi, matcher);
    
    alg.Compute();
    
    //fx_timer_stop(NULL, "single_bandwidth_time");
    IO::StopTimer("single_bandwidth_time");
    
    //std::cout << "\nSingle Bandwidth num tuples: " << single_alg.num_tuples() << "\n\n";
    IO::Info << std::endl << "Single bandwidth num tuples: " <<  matcher.results();
    
    IO::Info << std::endl << std::endl;
    
  } // single bandwidth
  
  //if (fx_param_exists(NULL, "do_naive")) {
  //if (IO::HasParam("do_naive")) {
  if (IO::GetParam<bool>("do_naive")) {
    //std::cout << "Doing naive.\n";
    
    IO::Info << "Doing naive." << std::endl;
    
    //fx_timer_start(NULL, "naive_time");
    IO::StartTimer("naive_time");
    
    IO::GetParam<int>("tree/leaf_size") = std::max(data_mat.n_cols, 
                                                   random_mat.n_cols);
    
    std::vector<size_t> old_from_new_data;
    std::vector<size_t> old_from_new_random;

    // build the trees
    //NptNode* naive_data_tree = new NptNode(data_mat);
    //NptNode* naive_random_tree = new NptNode(random_mat);
    
    NptNode* naive_data_tree = new NptNode(data_mat, old_from_new_data);
    NptNode* naive_random_tree = new NptNode(random_mat, old_from_new_random);
    
    std::vector<std::vector<size_t>*> old_from_new_list(tuple_size);
    
    for (int i = 0; i < num_random; i++) {
      old_from_new_list[i] = &old_from_new_random;
    }
    for (int i = num_random; i < tuple_size; i++) {
      old_from_new_list[i] = &old_from_new_data;
    }
    
    
    comp_trees[0] = naive_random_tree;
    comp_trees[1] = naive_data_tree;
    
    SingleMatcher matcher(comp_mats, comp_weights, 
                          old_from_new_list,
                          matcher_lower_bounds,
                          matcher_upper_bounds);
    
    GenericNptAlg<SingleMatcher> alg(comp_trees, comp_multi, matcher, true);
    
    alg.Compute();
    
    
    //fx_timer_stop(NULL, "naive_time");
    IO::StopTimer("naive_time");
    
    //std::cout << "\nNaive num tuples: " << naive_alg.num_tuples() << "\n\n";
    IO::Info << std::endl << "Naive num tuples: " << matcher.results();
    
    IO::Info << std::endl << std::endl;
    
  } // do naive
  
  
  
  
  /*
  //if (fx_param_exists(NULL, "do_perm_free")) {
  if (IO::HasParam("do_perm_free")) {
    
    IO::Info << "Doing permutation free.\n";

    //fx_timer_start(NULL, "perm_free_time");
    IO::StartTimer("perm_free_time");
    
    PermFreeAlg alg(data_mat, weights, random_mat, random_weights, leaf_size, 
                    matcher_dists, bandwidth);
    
    alg.Compute();
    
    //fx_timer_stop(NULL, "perm_free_time");
    IO::StopTimer("perm_free_time");
    
    IO::Info << "\nPerm Free num tuples: " << std::endl;
    alg.print_num_tuples();
    IO::Info << std::endl << std::endl;
    
    
  } // perm free
  */

  //fx_done(NULL);
  
  return 0;
  
} // main()
