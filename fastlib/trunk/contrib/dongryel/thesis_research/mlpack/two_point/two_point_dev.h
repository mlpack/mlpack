/*
 *  two_point_dev.h
 *  
 *
 *  Created by William March on 9/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TWO_POINT_DEV_H
#define TWO_POINT_DEV_H

#include "core/gnp/dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/transform.h"
#include "mlpack/two_point/two_point.h"


namespace mlpack {
  namespace two_point {

  bool TwoPointArgumentParser::ConstructBoostVariableMap(
    int argc,
    char *argv[],
    boost::program_options::variables_map *vm) {

    // Convert C input to C++; skip executable name for Boost.
    std::vector<std::string> args(argv + 1, argv + argc);
    
    // Call the other function.
    return ConstructBoostVariableMap(args, vm);
  }
    
  bool TwoPointArgumentParser::ConstructBoostVariableMap(
    const std::vector<std::string> &args,
    boost::program_options::variables_map *vm) 
  {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "data_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing the point set."
  )(
    "randoms_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing random (Poisson) positions.  If omitted, "
    "then the DD count is computed."
  )(
    "counts_out",
    boost::program_options::value<std::string>()->default_value(
      "counts_out.csv"),
    "OPTIONAL file to store computed count."
  )(
    "matcher_distance",
    boost::program_options::value<double>(),
    "REQUIRED Scale of the correlation (r in DD(r))."
  )(
    "matcher_thickness",
    boost::program_options::value<double>(),
    "REQUIRED Thickness of the matcher (Delta r)."
    )(
    "leaf_size",
    boost::program_options::value<int>()->default_value(20),
    "Maximum number of points at a leaf of the tree."
  );
    
  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), *vm);
  }
  catch(const boost::program_options::invalid_option_value &e) {
    std::cerr << "Invalid Argument: " << e.what() << "\n";
    exit(0);
  }
  catch(const boost::program_options::invalid_command_line_syntax &e) {
    std::cerr << "Invalid command line syntax: " << e.what() << "\n";
    exit(0);
  }
  catch(const boost::program_options::unknown_option &e) {
    std::cerr << "Unknown option: " << e.what() << "\n";
    exit(0);
  }

  boost::program_options::notify(*vm);
  if(vm->count("help")) {
    std::cout << desc << "\n";
    return true;
  }

  // Validate the arguments. Only immediate termination is allowed
  // here, the parsing is done later.
  if(vm->count("data_in") == 0) {
    std::cerr << "Missing required --data_in.\n";
    exit(0);
  }
  if((*vm)["leaf_size"].as<int>() <= 0) {
    std::cerr << "The --leaf_size needs to be a positive integer.\n";
    exit(0);
  }
  if ((*vm)["matcher_distance"].as<double>() <= 0.0) {
    std::cerr << "The --matcher_distance needs to be a positive real.\n";
    exit(0);    
  }
  if ((*vm)["matcher_thickness"].as<double>() <= 0.0) {
    std::cerr << "The --matcher_thickness needs to be a positive real.\n";
    exit(0);        
  }
  return false;
}


template<typename TableType>
bool TwoPointArgumentParser::ParseArguments(
         boost::program_options::variables_map &vm,
         TwoPointArguments<TableType> *arguments_out) {

  // A L2 metric to index the table to use.
  arguments_out->metric_ = new core::metric_kernels::LMetric<2>();
  
  // Given the constructed boost variable map, parse each argument.
  
  // Parse the densities out file.
  arguments_out->counts_out = vm["counts_out"].as<std::string>();
  
  // Parse the leaf size.
  arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
  std::cout << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";
  
  // Parse the reference set and index the tree.
  std::cout << "Reading in the data set: " <<
  vm["data_in"].as<std::string>() << "\n";
  arguments_out->points_table_1_ = new TableType();
  arguments_out->points_table_1_->Init(vm["data_in"].as<std::string>());
  std::cout << "Finished reading in the data set.\n";
  
  // Scale the dataset.
  std::cout << "Building the reference tree.\n";
  arguments_out->points_table_1_->IndexData(
       *(arguments_out->metric_), arguments_out->leaf_size_);
  std::cout << "Finished building the data tree.\n";
  
  // Parse the query set and index the tree.
  if(vm.count("randoms_in") > 0) {
    std::cout << "Reading in the random set: " <<
    vm["randoms_in"].as<std::string>() << "\n";
    arguments_out->points_table_2_ = new TableType();
    arguments_out->points_table_2_->Init(vm["randoms_in"].as<std::string>());
    std::cout << "Finished reading in the random set.\n";
    
    std::cout << "Building the query tree.\n";
    arguments_out->points_table_2_->IndexData(
               *(arguments_out->metric_), arguments_out->leaf_size_);
    std::cout << "Finished building the query tree.\n";
  }
  else {
    arguments_out->points_table_2_ = arguments_out->points_table_1_;
  }
  
  // Parse the bandwidth.
  arguments_out->matcher_distance_ = vm["matcher_distance"].as<double>();
  std::cout << "Matcher distance of " << arguments_out->matcher_distance_ << "\n";
  
  arguments_out->matcher_thickness_ = vm["matcher_thickness"].as<double>();
  std::cout << "Matcher thickness of " << arguments_out->matcher_thickness_ << "\n";

  return false;
} // ParseArguments
    
template<typename TableType>
template<typename IncomingGlobalType>
void TwoPoint<TableType>::Init(
     TwoPointArguments<TableType> &arguments_in, IncomingGlobalType *global_in) {
  
  // NOTE: the global_in is coming in NULL -- why?
  
  points_table_1_ = arguments_in.points_table_1_;
  if(arguments_in.points_table_1_ == arguments_in.points_table_2_) {
    is_monochromatic_ = true;
    points_table_2_ = points_table_1_;
  }
  else {
    is_monochromatic_ = false;
    points_table_2_ = arguments_in.points_table_2_;
  }
  
  // Declare the global constants.
  global_.Init(points_table_1_, points_table_2_,
               arguments_in.matcher_distance_,
               arguments_in.matcher_thickness_, 
               is_monochromatic_);
} // Init
    
    
template<typename TableType>
void TwoPoint<TableType>::Compute(
    const TwoPointArguments<TableType> &arguments_in,
    ResultType *result_out) {

  // Instantiate a dual-tree algorithm of the TwoPoint.
  typedef TwoPoint<TableType> ProblemType;
  core::gnp::DualtreeDfs< ProblemType > dualtree_dfs;
  dualtree_dfs.Init(*this);
  
  dualtree_dfs.Compute(* arguments_in.metric_, result_out);
  printf("Number of prunes: %d\n", dualtree_dfs.num_deterministic_prunes());
  printf("Number of probabilistic prunes: %d\n",
         dualtree_dfs.num_probabilistic_prunes());
  
  //printf("Number of tuples: %d\n", result_out->num_tuples());
  printf("Number of tuples: %d\n", result_out->num_tuples_);
  
}



    
  } // namespace
} // namespace


#endif

