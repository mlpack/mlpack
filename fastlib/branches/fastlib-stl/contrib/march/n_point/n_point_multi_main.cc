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
#include "matcher_generation.h"

using namespace npt;

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);
  
  // read in data and parameters
  
  std::string data_filename = fx_param_str(NULL, "data", "test_npt_pts.csv");
  arma::mat data_in, data_mat;
  data_in.load(data_filename, arma::raw_ascii);
  
  if (data_in.n_rows > data_in.n_cols) {
    data_mat = arma::trans(data_in);
  }
  else {
    data_mat = data_in;
  }
  // delete data_in
  
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
  
  // input format: each row is a pair (min, max, num)

  double bandwidth = fx_param_double(NULL, "bandwidth", 0.05);
  
  std::string matcher_filename = fx_param_str(NULL, "matchers",
                                              "test_matchers.csv");
  
  arma::mat matcher_mat;
  matcher_mat.load(matcher_filename, arma::raw_ascii);

  std::vector<double> min_bands(matcher_mat.n_cols);
  std::vector<double> max_bands(matcher_mat.n_cols);
  std::vector<int> num_bands(matcher_mat.n_cols);
  
  for (index_t i = 0; i < matcher_mat.n_rows; i++) {
   
    min_bands[i] = matcher_mat.at(i, 0);
    max_bands[i] = matcher_mat.at(i, 1);
    num_bands[i] = (int)matcher_mat.at(i,2);
    
  }

  int tuple_size = (1 + (int)sqrt(1 + 8 * num_bands.size())) / 2;
  //std::cout << "tuple size: " << tuple_size << "\n";
  
  MatcherGenerator generator(min_bands, max_bands, num_bands, tuple_size);
  
  //generator.print();
  
  // run algorithm
  
  
  
  if (fx_param_exists(NULL, "do_naive")) {
  
    std::cout << "\nDoing naive.\n";
    
    fx_timer_start(NULL, "naive_time");
    
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      NaiveAlg naive_alg(data_mat, weights, generator.matcher(i), bandwidth);
      
      naive_alg.ComputeCounts();
      
      generator.matcher(i).print("Matcher: ");
      std::cout << "Naive num tuples: " << naive_alg.num_tuples() << "\n\n";
    
    }
    
    fx_timer_stop(NULL, "naive_time");
    
  } // do naive
  
  
  index_t leaf_size = fx_param_int(NULL, "leaf_size", 1);
  
  
  if (fx_param_exists(NULL, "do_single_bandwidth")) {
    
    std::cout << "\nDoing single bandwidth.\n";

    fx_timer_start(NULL, "single_bandwidth_time");
    
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      
    
    
      SingleBandwidthAlg single_alg(data_mat, weights, leaf_size, 
                                    generator.matcher(i), bandwidth);
      
      single_alg.ComputeCounts();

      
      generator.matcher(i).print("Matcher: ");
      std::cout << "Single Bandwidth num tuples: " << single_alg.num_tuples() << "\n\n";

    }
      
    fx_timer_stop(NULL, "single_bandwidth_time");
    
    
  }
  
  
  
  if (fx_param_exists(NULL, "do_perm_free")) {
    
    std::cout << "\nDoing permutation free.\n";

    fx_timer_start(NULL, "perm_free_time");
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      PermFreeAlg alg(data_mat, weights, leaf_size, generator.matcher(i), 
                      bandwidth);
      
      alg.Compute();
      
      generator.matcher(i).print("Matcher: ");
      std::cout << "\nPerm Free num tuples: " << alg.num_tuples() << "\n\n";
      
    }
    
    fx_timer_stop(NULL, "perm_free_time");
    
  } // perm free
  

  if (fx_param_exists(NULL, "do_multi")) {
    
    std::cout << "\nDoing Multi Bandwidth\n";

        
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