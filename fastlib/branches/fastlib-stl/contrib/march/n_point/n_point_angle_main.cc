/*
 *  n_point_angle_main.cc
 *  
 *
 *  Created by William March on 8/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#include "angle_3pt_alg.h"
#include "angle_matcher_generation.h"
#include "single_bandwidth_alg.h"
#include "fastlib/fastlib.h"

PARAM_STRING_REQ("data", "Point coordinates.", NULL);
PARAM_STRING_REQ("random", "Poisson set coordinates.", NULL);
PARAM_FLAG("weighted_computation", "Specify if computing with pointwise weights", NULL)
PARAM_STRING("weights", "Optional data weights.", NULL, "default_weights.csv");
PARAM_STRING("random_weights", "Optional weights on Poisson set.", NULL, "default_weights.csv");
PARAM_STRING_REQ("matchers", "A 3 column, 2 row csv, row 1 is r1_min, r1_max, num_r1, row2 is the same for theta",
                 NULL);
PARAM_DOUBLE("bin_thickness_factor", "The multiplier to determine the thickness of each bin.  Delta_r in the literature.", NULL,
             1.0)
PARAM_DOUBLE("r2_multiplier", "How much larger is r2 than r1.", NULL, 2.0)
PARAM_INT("leaf_size", "Max number of points in a leaf node", NULL, 
          1);
//PARAM_FLAG("do_naive", "Permform Naive computation", NULL);
PARAM_FLAG("do_single_bandwidth", "Permform old (Moore & Gray) tree computation", NULL);
//PARAM_FLAG("do_perm_free", "Tree computation with alternative pruning rule", NULL);
//PARAM_FLAG("do_multi", "Multi-bandwidth computation -- i.e. one pass through the tree.", NULL);
PARAM_FLAG("do_angle", "Specialized angle-based computation.", NULL);

using namespace npt;
using namespace mlpack;


int main(int argc, char* argv[]) {

  IO::ParseCommandLine(argc, argv);
  
  
  std::string data_filename = IO::GetParam<std::string>("data");
  arma::mat data_in, data_mat;
  data_in.load(data_filename, arma::raw_ascii);
  
  if (data_in.n_rows > data_in.n_cols) {
    data_mat = arma::trans(data_in);
  }
  else {
    data_mat = data_in;
  }
  // delete data_in

  arma::colvec weights;  
  //if (fx_param_exists(NULL, "weights")) {
  if (IO::HasParam("weighted_computation")) {
    weights.load(IO::GetParam<std::string>("weights"));
  }
  else {
    weights.set_size(data_mat.n_cols);
    weights.fill(1.0);
  }
  
  std::string random_filename = IO::GetParam<std::string>("random");
  
  arma::mat random_in, random_mat;
  random_in.load(random_filename, arma::raw_ascii);
  
  // THIS IS BAD: do it better
  if (random_in.n_rows > random_mat.n_cols) {
    random_mat = arma::trans(random_in);
  }
  else {
    random_mat = random_in;
  }
  
  //arma::mat data_out = arma::trans(data_mat);
  //data_out.save("3pt_test_data.csv", arma::raw_ascii);
  
  arma::colvec random_weights;  
  //if (fx_param_exists(NULL, "weights")) {
  if (IO::HasParam("weighted_computation")) {
    random_weights.load(IO::GetParam<std::string>("random_weights"));
  }
  else {
    random_weights.set_size(random_mat.n_cols);
    random_weights.fill(1.0);
  }
  
  double bin_thickness_factor = IO::GetParam<double>("bin_thickness_factor");
  double r2_multiplier = IO::GetParam<double>("r2_multiplier");
  
  index_t leaf_size = (index_t)IO::GetParam<int>("leaf_size");
  
  std::string matcher_filename = IO::GetParam<std::string>("matchers");
  arma::mat matcher_mat;
  matcher_mat.load(matcher_filename, arma::raw_ascii);

  
  // matcher_mat has row 1: r1_min, r1_max, num_r1
  // row 2: theta_min, theta_max, num_theta
  double r1_min = matcher_mat.at(0,0);
  double r1_max = matcher_mat.at(0,1);
  int num_r1 = (int)matcher_mat.at(0,2);
  
  double theta_min = matcher_mat.at(1,0);
  double theta_max = matcher_mat.at(1,1);
  int num_theta = (int)matcher_mat.at(1,2);
  
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
  
  
  // do multi-angle alg
  if (IO::HasParam("do_angle")) {
  
    IO::StartTimer("angle_time");
    
    Angle3ptAlg alg(data_mat, weights, random_mat, random_weights, leaf_size,
                    r1_vec, r2_multiplier, theta_vec, bin_thickness_factor);
    
    alg.Compute();
    
    // output the results
    IO::Info << "3pt Angle Results: \n";
    alg.OutputResults();
    IO::Info << "\n\n";
    
    IO::StopTimer("angle_time");
  
  } // do angle
  
  // do single bandwidth alg
  if (IO::HasParam("do_single_bandwidth")) {
    
    AngleMatcherGenerator generator(r1_vec, r2_multiplier, theta_vec, 
                                    bin_thickness_factor);
    
    IO::Info << "Single Bandwidth Results: \n";
    
    IO::StartTimer("single_bandwidth_time");
    
    // iterate over the matchers
    for (int r1_ind = 0; r1_ind < num_r1; r1_ind++) {
    
      for (int theta_ind = 0; theta_ind < num_theta; theta_ind++) {
        
        arma::mat lower_bounds = generator.lower_matcher(r1_ind, theta_ind);
        arma::mat upper_bounds = generator.upper_matcher(r1_ind, theta_ind);
        
        //lower_bounds.print("lower bound mat");
        //upper_bounds.print("upper bound mat");
        
        SingleBandwidthAlg single_alg(data_mat, weights, random_mat, 
                                      random_weights,
                                      leaf_size, lower_bounds, upper_bounds);
  
  
        single_alg.ComputeCounts();
        
        IO::Info << "R1 = " << r1_vec[r1_ind] << ", ";
        IO::Info << "Theta = " << theta_vec[theta_ind] << ", ";
        IO::Info << "Single alg num_tuples: \n";
        
        single_alg.print_num_tuples();
        
        IO::Info << "\n";
        
        
      } // for theta_ind
      
    } // for r1_ind
    
    IO::StopTimer("single_bandwidth_time");
    
    IO::Info << "\n\n";
    
  } // doing single bandwidth
  
  
  return 0;
  
}