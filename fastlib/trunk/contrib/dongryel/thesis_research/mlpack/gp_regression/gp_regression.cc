/** @file gp_regression.cc
 *
 *  The main driver for the Gaussian process regression.
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
#include "mlpack/gp_regression/gp_regression_argument_parser.h"
#include "mlpack/gp_regression/gp_regression_dev.h"

template<typename KernelType, typename MetricType>
void StartComputation(boost::program_options::variables_map &vm) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree <
  mlpack::gp_regression::GpRegressionStatistic > > TableType;

  // Parse arguments for GP regression.
  mlpack::gp_regression::GpRegressionArguments <
  TableType, MetricType > gp_regression_arguments;
  if(mlpack::gp_regression::
      GpRegressionArgumentParser::ParseArguments(
        vm, &gp_regression_arguments)) {
    return;
  }

  // Instantiate a GP regression object.
  core::util::Timer init_timer;
  init_timer.Start();
  mlpack::gp_regression::GpRegression <
  TableType, KernelType, MetricType >
  gp_regression_instance;
  gp_regression_instance.Init(
    gp_regression_arguments,
    (typename mlpack::gp_regression::GpRegression <
     TableType, KernelType, MetricType >::GlobalType *) NULL);
  init_timer.End();
  std::cerr << init_timer.GetTotalElapsedTime() <<
            " seconds elapsed in initializing...\n";

  // Compute the result.
  core::util::Timer compute_timer;
  compute_timer.Start();
  mlpack::gp_regression::GpRegressionResult gp_regression_result;
  gp_regression_instance.Compute(
    gp_regression_arguments, &gp_regression_result);
  compute_timer.End();
  std::cerr << compute_timer.GetTotalElapsedTime() <<
            " seconds elapsed in computation...\n";

  // Output the GP regression result to the file.
  std::cerr << "Writing the predictions to the file: " <<
            gp_regression_arguments.predictions_out_ << "\n";
  gp_regression_result.Print(
    gp_regression_arguments.predictions_out_);
}

int main(int argc, char *argv[]) {

  boost::program_options::variables_map vm;
  if(mlpack::gp_regression::GpRegressionArgumentParser::
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

  return 0;
}
