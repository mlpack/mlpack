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
using namespace mlpack;

int main(int argc, char* argv[]) {

  IO::ParseCommandLine(argc, argv);
  
  // read in data and parameters
  
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
  
  //arma::mat data_out = arma::trans(data_mat);
  //data_out.save("3pt_test_data.csv", arma::raw_ascii);
  
  arma::colvec weights;  
  if (IO::HasParam("weights")) {
    weights.load(IO::GetParam<std::string>("weights"));
  }
  else {
    weights.set_size(data_mat.n_cols);
    weights.fill(1.0);
  }
  
  // input format: each row is a pair (min, max, num)

  double bandwidth = IO::GetParam<double>("bandwidth");
  
  //std::string matcher_filename = fx_param_str(NULL, "matchers",
  //                                            "test_matchers.csv");
  std::string matcher_filename = IO::GetParam<std::string>("matchers");
  
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
  
  
  
  if (IO::HasParam("do_naive")) {
    
    IO::Info << "\nDoing naive.\n";
    
    IO::StartTimer("naive_time");
    
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      NaiveAlg naive_alg(data_mat, weights, generator.matcher(i), bandwidth);
      
      naive_alg.ComputeCounts();
      
      generator.matcher(i).print("Matcher: ");
      IO::Info << "Naive num tuples: " << naive_alg.num_tuples() << "\n\n";
    
    }
    
    IO::StopTimer("naive_time");
    
  } // do naive
  
  
  index_t leaf_size = IO::GetParam<index_t>("leaf_size");
  
  
  if (IO::HasParam("do_single_bandwidth")) {
    
    IO::Info << "\nDoing single bandwidth.\n";

    IO::StartTimer("single_bandwidth_time");
    
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      
    
    
      SingleBandwidthAlg single_alg(data_mat, weights, leaf_size, 
                                    generator.matcher(i), bandwidth);
      
      single_alg.ComputeCounts();

      
      generator.matcher(i).print("Matcher: ");
      IO::Info << "Single Bandwidth num tuples: " << single_alg.num_tuples() << "\n\n";

    }
      
    IO::StopTimer("single_bandwidth_time");
    
    
  } // single bandwidth
  
  
  
  if (IO::HasParam("do_perm_free")) {
    
    IO::Info << "\nDoing permutation free.\n";

    IO::StartTimer("perm_free_time");
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      PermFreeAlg alg(data_mat, weights, leaf_size, generator.matcher(i), 
                      bandwidth);
      
      alg.Compute();
      
      generator.matcher(i).print("Matcher: ");
      IO::Info << "\nPerm Free num tuples: " << alg.num_tuples() << "\n\n";
      
    }
    
    IO::StopTimer("perm_free_time");
    
  } // perm free
  

  if (IO::HasParam("do_multi")) {
    
    IO::Info << "\nDoing Multi Bandwidth\n";

        
    IO::StartTimer("multi_time");
    
    MultiBandwidthAlg alg(data_mat, weights, leaf_size, tuple_size,
                          min_bands, max_bands, num_bands, bandwidth);
    
    alg.Compute();
    
    alg.OutputResults();

    IO::StopTimer("multi_time");
    
  } // multi

  
  
  return 0;
  
} // main()