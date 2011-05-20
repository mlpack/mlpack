/** @file local_regression.cc
 *
 *  The main driver for the local regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <armadillo>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "core/util/timer.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/local_regression/local_regression_argument_parser.h"
#include "mlpack/local_regression/local_regression_dev.h"
#include "mlpack/local_regression/local_regression_result.h"
#include "mlpack/series_expansion/kernel_aux.h"

template<typename KernelAuxType>
void StartComputation(boost::program_options::variables_map &vm) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree <
  core::tree::AbstractStatistic > > TableType;

  // Parse arguments for local regression.
  mlpack::local_regression::LocalRegressionArguments <
  TableType, core::metric_kernels::LMetric<2> > local_regression_arguments;
  if(mlpack::local_regression::
      LocalRegressionArgumentParser::ParseArguments(
        vm, &local_regression_arguments)) {
    return;
  }

  // Instantiate a local regression object.
  core::util::Timer init_timer;
  init_timer.Start();
  mlpack::local_regression::LocalRegression <
  TableType, KernelAuxType,  core::metric_kernels::LMetric<2> >
  local_regression_instance;
  local_regression_instance.Init(
    local_regression_arguments);
  init_timer.End();
  std::cerr << init_timer.GetTotalElapsedTime() <<
            " seconds elapsed in initializing...\n";

  // Compute the result.
  core::util::Timer compute_timer;
  compute_timer.Start();
  mlpack::local_regression::LocalRegressionResult local_regression_result;
  local_regression_instance.Compute(
    local_regression_arguments, &local_regression_result);
  compute_timer.End();
  std::cerr << compute_timer.GetTotalElapsedTime() <<
            " seconds elapsed in computation...\n";

  // Output the local regression result to the file.
  std::cerr << "Writing the predictions to the file: " <<
            local_regression_arguments.predictions_out_ << "\n";
  //local_regression_result.Print(
  //local_regression_arguments.predictions_out_);
}

int main(int argc, char *argv[]) {

  boost::program_options::variables_map vm;
  if(mlpack::local_regression::LocalRegressionArgumentParser::
      ConstructBoostVariableMap(
        argc, argv, &vm)) {
    return 0;
  }

  // Do a quick peek at the kernel and expansion type.
  std::string kernel_type = vm["kernel"].as<std::string>();
  std::string series_expansion_type =
    vm["series_expansion_type"].as<std::string>();

  if(kernel_type == "gaussian") {
    if(series_expansion_type == "hypercube") {
      StartComputation <
      mlpack::series_expansion::GaussianKernelHypercubeAux > (vm);
    }
    else {
      StartComputation <
      mlpack::series_expansion::GaussianKernelMultivariateAux > (vm);
    }
  }
  else {

    // Only the multivariate expansion is available for the
    // Epanechnikov.
    StartComputation <
    mlpack::series_expansion::EpanKernelMultivariateAux > (vm);
  }

  return 0;
}
