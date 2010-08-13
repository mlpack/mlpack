/** @file clusterwise_regression_defs.h
 *
 *  @brief A parameter specification for clusterwise regression via
 *         the EM algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEFS_H
#define MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEFS_H

#include <iostream>
#include "boost/program_options.hpp"
#include "boost/lexical_cast.hpp"
#include "fastlib/fastlib.h"
#include "clusterwise_regression.h"

int fl::ml::ClusterwiseRegression::RunAlgorithm(
  boost::program_options::variables_map &vm) {

  // Read the reference set.
  Matrix reference_set;
  data::Load(vm["references_in"].as<std::string>().c_str(), &reference_set);

  // Read the target set.
  Matrix target_set;
  Vector target_set_alias;
  data::Load(vm["targets_in"].as<std::string>().c_str(), &target_set);
  target_set_alias.Init(target_set.n_cols());
  for (int i = 0; i < target_set.n_cols(); i++) {
    target_set_alias[i] = target_set.get(0, i);
  }

  // The algorithm object and its result.
  fl::ml::ClusterwiseRegression algorithm;
  fl::ml::ClusterwiseRegressionResult result;
  algorithm.Init(reference_set, target_set_alias);

  // Compute and export.
  int num_iterations = (vm.count("num_iterations") > 0) ?
                       vm["num_iterations"].as<int>() : 1000;
  algorithm.Compute(vm["k_clusters"].as<int>(), num_iterations, &result);

  return 0;
}

int fl::ml::ClusterwiseRegression::Main(
  const std::vector<std::string> &args) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "k_clusters",
    boost::program_options::value<int>(),
    "REQUIRED the number of clusters in the mixture of experts model")
  (
    "num_iterations",
    boost::program_options::value<int>(),
    "OPTIONAL the maximum number of iterations to train the mixture "
    "of experts")
  (
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data"
  )(
    "targets_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing the targets"
  )(
    "loglevel",
    boost::program_options::value<std::string>()->default_value("debug"),
    "Level of log detail.  One of:\n"
    "  debug: log everything\n"
    "  verbose: log messages and warnings\n"
    "  warning: log only warnings\n"
    "  silent: no logging"
  );

  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), vm);
  }
  catch (const boost::program_options::invalid_option_value &e) {
    std::cerr << "Invalid Argument: " << e.what();
    throw new std::exception();
  }
  catch (const boost::program_options::invalid_command_line_syntax &e) {
    std::cerr << "Invalid command line syntax: " << e.what();
    throw new std::exception();
  }
  catch (const boost::program_options::unknown_option &e) {
    std::cerr << "Unknown option: " << e.what() ;
    throw new std::exception();
  }

  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return true;
  }

  // Validate the arguments.
  if (vm.count("k_clusters") == 0) {
    std::cerr << "Missing required --k_clusters";
    throw new std::exception();
  }
  if (vm.count("references_in") == 0) {
    std::cerr << "Missing required --references_in";
    throw new std::exception();
  }
  if (vm.count("targets_in") == 0) {
    std::cerr << "Missing required --targets_in";
    throw new std::exception();
  }

  return RunAlgorithm(vm);
}

#endif
