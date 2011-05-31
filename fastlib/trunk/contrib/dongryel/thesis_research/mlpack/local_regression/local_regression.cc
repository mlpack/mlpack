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
#include "core/metric_kernels/kernel.h"
#include "core/metric_kernels/lmetric.h"
#include "core/metric_kernels/weighted_lmetric.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/local_regression/local_regression_argument_parser.h"
#include "mlpack/local_regression/local_regression_dev.h"

template<typename KernelType, typename MetricType>
void StartComputation(boost::program_options::variables_map &vm) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree <
  mlpack::local_regression::LocalRegressionStatistic > > TableType;

  // Parse arguments for local regression.
  mlpack::local_regression::LocalRegressionArguments <
  TableType, MetricType > local_regression_arguments;
  if(mlpack::local_regression::
      LocalRegressionArgumentParser::ParseArguments(
        vm, &local_regression_arguments)) {
    return;
  }

  // Instantiate a local regression object.
  core::util::Timer init_timer;
  init_timer.Start();
  mlpack::local_regression::LocalRegression <
  TableType, KernelType, MetricType >
  local_regression_instance;
  local_regression_instance.Init(
    local_regression_arguments,
    (typename mlpack::local_regression::LocalRegression <
     TableType, KernelType, MetricType >::GlobalType *) NULL);
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
  local_regression_result.Print(
    local_regression_arguments.predictions_out_);
}

int main(int argc, char *argv[]) {

  boost::program_options::variables_map vm;
  if(mlpack::local_regression::LocalRegressionArgumentParser::
      ConstructBoostVariableMap(
        argc, argv, &vm)) {
    return 0;
  }

  // Do a quick peek at the kernel type.
  std::string kernel_type = vm["kernel"].as<std::string>();

  // Do a quick peek at the metric type.
  std::string metric_type = vm["metric"].as<std::string>();

  if(kernel_type == "gaussian") {
    if(metric_type == "weighted_lmetric") {
      StartComputation <
      core::metric_kernels::GaussianKernel,
           core::metric_kernels::WeightedLMetric<2> > (vm);
    }
    else {
      StartComputation <
      core::metric_kernels::GaussianKernel,
           core::metric_kernels::LMetric<2> > (vm);
    }
  }
  else {
    if(metric_type == "weighted_lmetric") {
      StartComputation <
      core::metric_kernels::EpanKernel,
           core::metric_kernels::WeightedLMetric<2> > (vm);
    }
    else {
      StartComputation <
      core::metric_kernels::EpanKernel,
           core::metric_kernels::LMetric<2> > (vm);
    }
  }

  return 0;
}
