/** @file gp_regression.cc
 *
 *  The main driver for the Gaussian process regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <iostream>
#include <string>
#include <armadillo>
#include "gp_regression_dev.h"
#include "core/tree/gen_metric_tree.h"

int main(int argc, char *argv[]) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::table::DensePoint> > TableType;

  // Parse arguments for GPR.
  mlpack::gp_regression::GpRegressionArguments<TableType> gpr_arguments;
  mlpack::gp_regression::GpRegression<TableType>::ParseArguments(
    argc, argv, &gpr_arguments);

  // Instantiate a GPR object.
  mlpack::gp_regression::GpRegression<TableType> gpr_instance;
  gpr_instance.Init(gpr_arguments);

  // Compute the result.
  mlpack::gp_regression::GpRegressionResult gpr_result;
  gpr_instance.Compute(gpr_arguments, &gpr_result);

  // Output the GPR result to the file.
  std::cerr << "Writing the regression values to the file: " <<
            gpr_arguments.predictions_out_ << "\n";
  gpr_result.PrintDebug(gpr_arguments.predictions_out_);

  return 0;
}
