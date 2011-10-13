/*
 *  test_single_bandwidth.cc
 *  
 *
 *  Created by William March on 10/4/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "test_single_bandwidth.h"

void npt::TestSingleBandwidth::GenerateRandomSet_(arma::mat& data) {
  
  for (unsigned int row_ind = 0; row_ind < data.n_rows; row_ind++) {
    
    for (unsigned int col_ind = 0; col_ind < data.n_cols; col_ind++) {
      
      data(row_ind,col_ind) = data_gen();
      
    }
    
  }
  
} // GenerateRandomSet();

double npt::TestSingleBandwidth::GenerateRandomMatcher_(arma::mat& matcher) {
  
  for (unsigned int i = 0; i < matcher.n_rows; i++) {
    
    matcher(i,i) = 0.0;
    
    for (unsigned int j = i+1; j < matcher.n_cols; j++) {
      
      matcher(i,j) = matcher_dist_gen();
      matcher(j,i) = matcher(i,j);
      
    }
    
  }
  
  return matcher_thick_gen();
  
}


bool npt::TestSingleBandwidth::StressTest() {
  
  // pick a number of points and number of dimensions (small)
  
  printf("generating data\n");
  int num_data_points = num_data_gen();
  printf("num_data: %d, generating random\n", num_data_points);
  int num_random_points = num_data_gen();
  printf("num_random: %d, generating tuple_size\n", num_random_points);
  int num_dimensions = 3;
  printf("generating tuple size\n");
  int tuple_size = tuple_size_gen();
  
  mlpack::Log::Info << "====Running Test===\n";
  mlpack::Log::Info << "Tuple size: " << tuple_size << "\n";
  mlpack::Log::Info << "Num data points: " << num_data_points << "\n";
  mlpack::Log::Info << "Num random points: " << num_random_points << "\n";
  
  
  // Generate a random data set
  
  printf("generating_data\n");
  arma::mat data_mat(num_dimensions, num_data_points);
  GenerateRandomSet_(data_mat);
  arma::colvec data_weights(num_data_points);
  
  // Generate a random random set
  printf("generating_random\n");
  arma::mat random_mat(num_dimensions, num_random_points);
  GenerateRandomSet_(random_mat);
  arma::colvec random_weights(num_random_points);
  printf("random generated\n");
  
  // Generate a random matcher and matcher thickness multiplier
  
  printf("generating matchers \n");
  arma::mat matcher_dists(tuple_size, tuple_size);
  double matcher_thick = GenerateRandomMatcher_(matcher_dists);
  
  // Run tree algorithm

  // build trees
  std::vector<size_t> old_from_new_data;
  std::vector<size_t> old_from_new_random;
  
  std::vector<size_t> naive_old_from_new_data;
  std::vector<size_t> naive_old_from_new_random;
  
  std::vector<int> tree_results(tuple_size+1);
  std::vector<int> naive_results(tuple_size+1);
  
  int leaf_size = num_leaves_gen();
  
  mlpack::IO::GetParam<int>("tree/leaf_size") = leaf_size;
  
  mlpack::Log::Info << "Leaf size: " << leaf_size << "\n";
  
  printf("building trees\n");
  NptNode* data_tree = new NptNode(data_mat, old_from_new_data);
  NptNode* random_tree = new NptNode(random_mat, old_from_new_random);
  
  mlpack::IO::GetParam<int>("tree/leaf_size") = std::max(num_data_points,
                                                 num_random_points);
  
  NptNode* naive_data_tree = new NptNode(data_mat, naive_old_from_new_data);
  NptNode* naive_random_tree = new NptNode(random_mat, 
                                           naive_old_from_new_random);
  
  
  bool results_match = true;
  
  printf("starting main loop\n");
  // iterate over num_random
  for (int num_random = 0; num_random <= tuple_size && results_match; 
       num_random++) {
    
    std::vector<arma::mat*> comp_mats(tuple_size);
    // only 2 entries, since we're not worried about cross correlations yet
    //std::vector<int> comp_multi(2, 0);
    std::vector<arma::colvec*> comp_weights(tuple_size);
    std::vector<NptNode*> comp_trees(tuple_size);
    std::vector<NptNode*> naive_comp_trees(tuple_size);
    std::vector<std::vector<size_t>*> old_from_new_list(tuple_size);
    std::vector<std::vector<size_t>*> naive_old_from_new_list(tuple_size);
    
    for (int i = 0; i < num_random; i++) {
      comp_mats[i] = &random_mat;
      comp_weights[i] = &random_weights;
      //comp_multi[0]++;
      comp_trees[i] = random_tree;
      naive_comp_trees[i] = naive_random_tree;
      old_from_new_list[i] = &old_from_new_random;
      naive_old_from_new_list[i] = &naive_old_from_new_random;
    }
    for (int i = num_random; i < tuple_size; i++) {
      comp_mats[i] = &data_mat;
      comp_weights[i] = &data_weights;
      //comp_multi[1]++;
      comp_trees[i] = data_tree;
      naive_comp_trees[i] = naive_data_tree;
      old_from_new_list[i] = &old_from_new_data;
      naive_old_from_new_list[i] = &naive_old_from_new_data;
    }
    
    //comp_trees[0] = random_tree;
    //comp_trees[1] = data_tree;
    
    //naive_comp_trees[0] = naive_random_tree;
    //naive_comp_trees[1] = naive_data_tree;
    
    printf("Running tree algorithm, iteration %d\n", num_random);
    SingleMatcher tree_matcher(comp_mats, comp_weights, old_from_new_list,
                               matcher_dists, matcher_thick);
  
    GenericNptAlg<SingleMatcher> tree_alg(comp_trees, 
                                          //comp_multi, 
                                          tree_matcher);
    
    tree_alg.Compute();
    
    // Get and store the result
    tree_results[num_random] = tree_matcher.results();
    
    mlpack::Log::Info << "tree_results[" << num_random << "]: ";
    mlpack::Log::Info << tree_results[num_random] << "\n";
    
    
    printf("Running naive algorithm, iteration %d\n", num_random);
    SingleMatcher naive_matcher(comp_mats, comp_weights, 
                                naive_old_from_new_list,
                                matcher_dists, matcher_thick);
    
    GenericNptAlg<SingleMatcher> naive_alg(naive_comp_trees, 
                                           //comp_multi,
                                           naive_matcher);
  
    naive_alg.Compute();
    
    naive_results[num_random] = naive_matcher.results();
    
    mlpack::Log::Info << "naive_results[" << num_random << "]: ";
    mlpack::Log::Info << naive_results[num_random] << "\n";

    
    // Compare results
    results_match = (naive_results[num_random] == tree_results[num_random]);
    
    if (!results_match) {
      
      //mlpack::Log::Info << "Results fail to match for num_random: ";
      //mlpack::Log::Info << num_random << "\n";
      printf("Results fail to match\n");
      
      
    }
    
    mlpack::Log::Info << "\n\n";
    
  } // for num_random
    
  return results_match;
  
  
} // StressTest


// the driver
bool npt::TestSingleBandwidth::StressTestMain() {
 
  printf("running stress test main\n");
  
  bool results_match = true;
  
  for (int i = 0; i < 5 && results_match; i++) {
   
    printf("Running test: %d\n", i);
    results_match = StressTest();
    
  }
  
  return results_match;
  
}


