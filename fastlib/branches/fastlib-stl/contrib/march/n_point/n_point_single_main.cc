/*
 *  n_point_main.cc
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "fastlib/fastlib.h"

#include "single_bandwidth_alg.h"
#include "naive_alg.h"
#include "perm_free_alg.h"
#include "multi_bandwidth_alg.h"

using namespace npt;

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);
  
  // read in data and parameters
  
  std::string data_filename = fx_param_str(NULL, "data", "test_npt_pts.csv");
  arma::mat data_mat;
  data_mat.load(data_filename, arma::raw_ascii);
  
  //arma::mat data_out = arma::trans(data_mat);
  //data_out.save("3pt_test_data.csv", arma::raw_ascii);
  
  arma::colvec weights;  
  if (fx_param_exists(NULL, "weights")) {
    weights.load(fx_param_str_req(NULL, "weights"));
  }
  else {
    weights.set_size(data_mat.n_cols);
    weights.fill(1.0);
  }
  
  //std::cout << "loaded weights\n";
  
  arma::mat lower_bds, upper_bds;
  upper_bds.load(fx_param_str(NULL, "upper_bounds", "test_upper_bds.csv"));
  lower_bds.load(fx_param_str(NULL, "lower_bounds", "test_lower_bds.csv"));

  //std::cout << "loaded bounds\n";
  
  
  // run algorithm
  
  if (fx_param_exists(NULL, "do_naive")) {
  
    std::cout << "Doing naive.\n";
    
    fx_timer_start(NULL, "naive_time");
    
    NaiveAlg naive_alg(data_mat, weights, lower_bds, upper_bds);
    
    naive_alg.ComputeCounts();
    
    fx_timer_stop(NULL, "naive_time");
    
    std::cout << "\nNaive num tuples: " << naive_alg.num_tuples() << "\n\n";
    
  }
  
  
  index_t leaf_size = fx_param_int(NULL, "leaf_size", 1);
  
  
  if (fx_param_exists(NULL, "do_single_bandwidth")) {
    
    std::cout << "Doing single bandwidth.\n";

    fx_timer_start(NULL, "single_bandwidth_time");
    
    SingleBandwidthAlg single_alg(data_mat, weights, leaf_size, 
                                  lower_bds, upper_bds);
    
    single_alg.ComputeCounts();
    
    fx_timer_stop(NULL, "single_bandwidth_time");
    
    std::cout << "\nSingle Bandwidth num tuples: " << single_alg.num_tuples() << "\n\n";
    
  }
  
  
  
  if (fx_param_exists(NULL, "do_perm_free")) {
    
    std::cout << "Doing permutation free.\n";

    fx_timer_start(NULL, "perm_free_time");
    
    PermFreeAlg alg(data_mat, weights, leaf_size, lower_bds, upper_bds);
    
    alg.Compute();
    
    fx_timer_stop(NULL, "perm_free_time");
    
    std::cout << "\nPerm Free num tuples: " << alg.num_tuples() << "\n\n";
    
  } // perm free
  

  if (fx_param_exists(NULL, "do_multi")) {
    
    std::cout << "Doing Multi Bandwidth\n";

    
    int tuple_size = 3;
    std::vector<double> min_bands(3, 0.5);
    std::vector<double> max_bands(3, 1.0);
    std::vector<int> num_bands(3, 5);
    double bandwidth = 0.05;
    
    
    
    fx_timer_start(NULL, "multi_time");
    MultiBandwidthAlg alg(data_mat, weights, leaf_size, tuple_size,
                          min_bands, max_bands, num_bands, bandwidth);
    
    
    alg.Compute();
    
    alg.OutputResults();

    fx_timer_stop(NULL, "multi_time");
    
  } // multi

  
  
  fx_done(NULL);
  
  return 0;
  
} // main()