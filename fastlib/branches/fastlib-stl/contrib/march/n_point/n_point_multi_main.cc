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


PARAM_STRING_REQ("data", "Point coordinates.", NULL);
PARAM_STRING_REQ("random", "Poisson set coordinates.", NULL);
PARAM_FLAG("weighted_computation", "Specify if computing with pointwise weights", NULL)
PARAM_STRING("weights", "Optional data weights.", NULL, "default_weights.csv");
PARAM_STRING("random_weights", "Optional weights on Poisson set.", NULL, "default_weights.csv");
PARAM_STRING_REQ("matchers", "A 3 column, (n choose 2) row csv, where the first and second columns are the upper and lower bounds and the third is the number of matchers.",
                 NULL);
PARAM_DOUBLE("bandwidth", "Thickness of the matcher", NULL,
      1.0)
PARAM_INT("leaf_size", "Max number of points in a leaf node", NULL, 
      1);
PARAM_FLAG("do_naive", "Permform Naive computation", NULL);
PARAM_FLAG("do_single_bandwidth", "Permform old (Moore & Gray) tree computation", NULL);
PARAM_FLAG("do_perm_free", "Tree computation with alternative pruning rule", NULL);
PARAM_FLAG("do_multi", "Multi-bandwidth computation -- i.e. one pass through the tree.", NULL);



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
    weights.load(IO::GetParam<std::string>("random_weights"));
  }
  else {
    random_weights.set_size(random_mat.n_cols);
    random_weights.fill(1.0);
  }
  
  
  
  // input format: each row is a pair (min, max, num)

  double bandwidth = IO::GetParam<double>("bandwidth");
  
  //std::string matcher_filename = fx_param_str(NULL, "matchers",
  //                                            "test_matchers.csv");
  std::string matcher_filename = IO::GetParam<std::string>("matchers");
  
  arma::mat matcher_mat;
  matcher_mat.load(matcher_filename, arma::raw_ascii);

  // matcher mat should always have 3 columns
  
  std::vector<double> min_bands(matcher_mat.n_rows);
  std::vector<double> max_bands(matcher_mat.n_rows);
  std::vector<int> num_bands(matcher_mat.n_rows);
  
  for (index_t i = 0; i < matcher_mat.n_rows; i++) {
   
    min_bands[i] = matcher_mat.at(i, 0);
    max_bands[i] = matcher_mat.at(i, 1);
    num_bands[i] = (int)matcher_mat.at(i,2);
    
  }

  // TODO: be less stupid about this
  int tuple_size = (1 + (int)sqrt(1 + 8 * num_bands.size())) / 2;
  //std::cout << "tuple size: " << tuple_size << "\n";
  
  MatcherGenerator generator(min_bands, max_bands, num_bands, tuple_size);
  
  //generator.print();
  
  // run algorithm
  
  if (IO::HasParam("do_naive")) {
    
    IO::Info << "\nDoing naive.\n";
    
    IO::StartTimer("naive_time");
    
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      NaiveAlg naive_alg(data_mat, weights, random_mat, random_weights, 
                         generator.matcher(i), bandwidth);
      
      naive_alg.ComputeCounts();
      
      generator.matcher(i).print("Matcher: ");
      IO::Info << std::endl << "Naive num tuples: " << std::endl;
      
      naive_alg.print_num_tuples();
      
      IO::Info << std::endl << std::endl;
      
    }
    
    IO::StopTimer("naive_time");
    
  } // do naive
  
  
  index_t leaf_size = (index_t)IO::GetParam<int>("leaf_size");
  
  
  if (IO::HasParam("do_single_bandwidth")) {
    
    IO::Info << "\nDoing single bandwidth.\n";

    IO::StartTimer("single_bandwidth_time");
    
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      
    
    
      SingleBandwidthAlg single_alg(data_mat, weights, random_mat, random_weights,
                                    leaf_size, 
                                    generator.matcher(i), bandwidth);
      
      single_alg.ComputeCounts();

      
      generator.matcher(i).print("Matcher: ");
      IO::Info << std::endl << "Single bandwidth num tuples: " << std::endl;
      
      single_alg.print_num_tuples();
      
      IO::Info << std::endl << std::endl;
      
    }
      
    IO::StopTimer("single_bandwidth_time");
    
    
  } // single bandwidth
  
  
  
  if (IO::HasParam("do_perm_free")) {
    
    IO::Info << "\nDoing permutation free.\n";

    IO::StartTimer("perm_free_time");
    
    for (index_t i = 0; i < generator.num_matchers(); i++) {
      
      PermFreeAlg alg(data_mat, weights, random_mat, random_weights, 
                      leaf_size, generator.matcher(i), 
                      bandwidth);
      
      alg.Compute();
      
      generator.matcher(i).print("Matcher: ");
      IO::Info << "\nPerm Free num tuples: " << std::endl;
      alg.print_num_tuples();
      IO::Info << std::endl << std::endl;
      
    }
    
    IO::StopTimer("perm_free_time");
    
  } // perm free
  

  if (IO::HasParam("do_multi")) {
    
    IO::Info << "\nDoing Multi Bandwidth\n";

        
    IO::StartTimer("multi_time");
    
    MultiBandwidthAlg alg(data_mat, weights, random_mat, random_weights, 
                          leaf_size, tuple_size,
                          min_bands, max_bands, num_bands, bandwidth);
    
    alg.Compute();
    
    alg.OutputResults();

    IO::StopTimer("multi_time");
    
  } // multi

  
  
  return 0;
  
} // main()