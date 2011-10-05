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
      
      data(row_ind,col_ind) = data_dist(generator_);
      
    }
    
  }
  
} // GenerateRandomSet();

double npt::TestSingleBandwidth::GenerateRandomMatcher_(arma::mat& matcher) {
  
  for (unsigned int i = 0; i < matcher.n_rows; i++) {
    
    matcher(i,i) = 0.0;
    
    for (unsigned int j = i+1; j < matcher.n_cols; j++) {
      
      matcher(i,j) = matcher_dist(generator_);
      matcher(j,i) = matcher(i,j);
      
    }
    
  }
  
  return matcher_thick_dist(generator_);
  
}


bool npt::TestSingleBandwidth::StressTest() {
  
  // pick a number of points and number of dimensions (small)
  int num_data_points = num_data_dist(generator_);
  int num_random_points = num_data_dist(generator_);
  int num_dimensions = 3;
  int tuple_size = tuple_size_dist(generator_);
  
  mlpack::IO::Info << "====Running Test===\n";
  mlpack::IO::Info << "Tuple size: " << tuple_size << "\n";
  mlpack::IO::Info << "Num data points: " << num_data_points << "\n";
  mlpack::IO::Info << "Num random points: " << num_random_points << "\n";
  
  
  // Generate a random data set
  
  arma::mat data_mat(num_dimensions, num_data_points);
  GenerateRandomSet_(data_mat);
  arma::colvec data_weights(num_data_points);
  
  // Generate a random random set

  arma::mat random_mat(num_dimensions, num_random_points);
  GenerateRandomSet_(random_mat);
  arma::colvec random_weights(num_random_points);
  
  // Generate a random matcher and matcher thickness multiplier
  
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
  
  int leaf_size = num_leaves_dist(generator_);
  
  mlpack::IO::GetParam<int>("tree/leaf_size") = leaf_size;
  
  mlpack::IO::Info << "Leaf size: " << leaf_size << "\n";
  
  
  NptNode* data_tree = new NptNode(data_mat, old_from_new_data);
  NptNode* random_tree = new NptNode(random_mat, old_from_new_random);
  
  mlpack::IO::GetParam<int>("tree/leaf_size") = std::max(num_data_points,
                                                 num_random_points);
  
  NptNode* naive_data_tree = new NptNode(data_mat, naive_old_from_new_data);
  NptNode* naive_random_tree = new NptNode(random_mat, 
                                           naive_old_from_new_random);
  
  
  bool results_match = true;
  
  // iterate over num_random
  for (int num_random = 0; num_random <= tuple_size && results_match; 
       num_random++) {
    
    std::vector<arma::mat*> comp_mats(tuple_size);
    // only 2 entries, since we're not worried about cross correlations yet
    std::vector<int> comp_multi(2, 0);
    std::vector<arma::colvec*> comp_weights(tuple_size);
    std::vector<NptNode*> comp_trees(2);
    std::vector<NptNode*> naive_comp_trees(2);
    std::vector<std::vector<size_t>*> old_from_new_list(tuple_size);
    std::vector<std::vector<size_t>*> naive_old_from_new_list(tuple_size);
    
    for (int i = 0; i < num_random; i++) {
      comp_mats[i] = &random_mat;
      comp_weights[i] = &random_weights;
      comp_multi[0]++;
      old_from_new_list[i] = &old_from_new_random;
      naive_old_from_new_list[i] = &naive_old_from_new_random;
    }
    for (int i = num_random; i < tuple_size; i++) {
      comp_mats[i] = &data_mat;
      comp_weights[i] = &data_weights;
      comp_multi[1]++;
      old_from_new_list[i] = &old_from_new_data;
      naive_old_from_new_list[i] = &naive_old_from_new_data;
    }
    
    comp_trees[0] = random_tree;
    comp_trees[1] = data_tree;
    
    naive_comp_trees[0] = naive_random_tree;
    naive_comp_trees[1] = naive_data_tree;
    
    SingleMatcher tree_matcher(comp_mats, comp_weights, old_from_new_list,
                               matcher_dists, matcher_thick);
  
    GenericNptAlg<SingleMatcher> tree_alg(comp_trees, comp_multi, tree_matcher);
    
    tree_alg.Compute();
    
    // Get and store the result
    tree_results[num_random] = tree_matcher.results();
    
    mlpack::IO::Info << "tree_results[" << num_random << "]: ";
    mlpack::IO::Info << tree_results[num_random] << "\n";
    
    
    SingleMatcher naive_matcher(comp_mats, comp_weights, 
                                naive_old_from_new_list,
                                matcher_dists, matcher_thick);
    
    GenericNptAlg<SingleMatcher> naive_alg(naive_comp_trees, comp_multi,
                                           naive_matcher);
  
    naive_alg.Compute();
    
    naive_results[num_random] = naive_matcher.results();
    
    mlpack::IO::Info << "naive_results[" << num_random << "]: ";
    mlpack::IO::Info << naive_results[num_random] << "\n";

    
    // Compare results
    results_match = (naive_results[num_random] == tree_results[num_random]);
    
    if (!results_match) {
      
      mlpack::IO::Info << "Results fail to match for num_random: ";
      mlpack::IO::Info << num_random << "\n";
      
    }
    
  } // for num_random
  
  return results_match;
  
  
} // StressTest


// the driver
bool npt::TestSingleBandwidth::StressTestMain() {
 
  bool results_match = true;
  
  for (int i = 0; i < 20 && !results_match; i++) {
   
    results_match = StressTest();
    
  }
  
  return results_match;
  
}


