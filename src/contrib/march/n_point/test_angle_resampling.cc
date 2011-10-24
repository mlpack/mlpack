/*
 *  test_angle_resampling.cc
 *  
 *
 *  Created by William March on 10/4/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "test_angle_resampling.h"



void npt::TestAngleResampling::GenerateRandomSet_(arma::mat& data) {
  
  for (unsigned int row_ind = 0; row_ind < data.n_rows; row_ind++) {
    
    for (unsigned int col_ind = 0; col_ind < data.n_cols; col_ind++) {
      
      data(row_ind,col_ind) = data_gen();
      
    }
    
  }
  
} // GenerateRandomSet();


bool npt::TestAngleResampling::StressTest() {
  
  // pick a number of points and number of dimensions (small)
  
  int num_data_points = num_data_gen();
  int num_random_points = num_data_gen();
  int num_dimensions = 3;
  int tuple_size = 3;
  
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
  
  mlpack::CLI::GetParam<int>("tree/leaf_size") = num_leaves_gen();
  
  printf("generating matchers \n");
  
  double r1_min = 0.1;
  double r1_max = 0.5;
  int num_r1 = num_r1_gen();
  printf("num_r1: %d\n", num_r1);
  
  double theta_min = 0.175;
  double theta_max = 2.97;
  int num_theta = num_theta_gen();
  printf("num_theta: %d\n", num_theta);
  
  double r2_multiplier = r2_mult_gen();
  
  double bin_thickness_factor = bin_thick_gen();
  
  int num_x_regions = num_regions_gen();
  int num_y_regions = num_regions_gen();
  int num_z_regions = num_regions_gen();
  
  printf("num_x: %d, num_y: %d, num_z: %d\n",
         num_x_regions, num_y_regions, num_z_regions);
  
  double box_length = 1.0;
  
  // print matcher info
  
  // dividing by zero if only one r1
  double r1_step = (r1_max - r1_min) / ((double)num_r1 - 1.0);
  
  std::vector<double> r1_vec(num_r1);
  
  if (num_r1 > 1) {
    for (int i = 0; i < num_r1; i++) {
      r1_vec[i] = r1_min + (double)i * r1_step;
    }
  }
  else {
    r1_vec[0] = r1_min;
  }
  
  double theta_step = (theta_max - theta_min) / ((double)num_theta - 1.0);
  
  std::vector<double> theta_vec(num_theta);
  
  if (num_theta > 1) {
    for (int i = 0; i < num_theta; i++) {
      theta_vec[i] = theta_min + (double)i * theta_step;
    }
  }
  else {
    theta_vec[0] = theta_min;
  }
  
  printf("Running multi angle alg.\n");
  AngleDriver alg(data_mat, data_weights, random_mat, random_weights,
                  r1_vec, r2_multiplier, theta_vec, bin_thickness_factor,
                  num_x_regions, num_y_regions, num_z_regions, 
                  box_length, box_length, box_length);
 
  alg.Compute();
  
  printf("Angle results: \n");
  alg.PrintResults();
  

  printf("Running single alg.\n");
  SingleDriver single_alg(data_mat, data_weights, random_mat, random_weights,
                          r1_vec, r2_multiplier, theta_vec, bin_thickness_factor,
                          num_x_regions, num_y_regions, num_z_regions, 
                          box_length, box_length, box_length);
  
  single_alg.Compute();
    
  printf("Single results: \n");
  single_alg.PrintResults();
  
  
  bool results_match = true;
  
  /*
  for (int i = 0; ; <#increment#>) {
    <#statements#>
  }
   */
  
  if (!results_match) {
  
    //mlpack::Log::Info << "Results fail to match for num_random: ";
    //mlpack::Log::Info << num_random << "\n";
    printf("Results fail to match\n");
    
  
  }

  return results_match;
  
  
} // StressTest


// the driver
bool npt::TestAngleResampling::StressTestMain() {
  
  printf("running stress test main\n");
  
  bool results_match = true;
  
  for (int i = 0; i < 5 && results_match; i++) {
    
    printf("Running test: %d\n", i);
    results_match = StressTest();
    
  }
  
  return results_match;
  
}

