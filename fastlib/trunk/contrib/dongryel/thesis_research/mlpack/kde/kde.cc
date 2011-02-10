/** @file kde.cc
 *
 *  The main driver for the KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <armadillo>
#include <iostream>
#include <string>
#include "core/tree/gen_metric_tree.h"
#include "mlpack/kde/kde_dev.h"
#include "mlpack/series_expansion/kernel_aux.h"

template<typename TableType, , typename KernelAuxType>
void StartComputation(mlpack::kde::KdeArguments<TableType> &kde_arguments) {

  // Instantiate a KDE object.
  mlpack::kde::Kde<TableType, KernelAuxType> kde_instance;
  kde_instance.Init(kde_arguments);

  // Compute the result.
  mlpack::kde::KdeResult< std::vector<double> > kde_result;
  kde_instance.Compute(kde_arguments, &kde_result);

  // Output the KDE result to the file.
  std::cerr << "Writing the densities to the file: " <<
            kde_arguments.densities_out_ << "\n";
  kde_result.Print(kde_arguments.densities_out_);
}

int main(int argc, char *argv[]) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<mlpack::kde::KdeStatistic> > TableType;

  // Parse arguments for Kde.
  mlpack::kde::KdeArguments<TableType> kde_arguments;
  if(mlpack::kde::Kde<TableType>::ParseArguments(argc, argv, &kde_arguments)) {
    return 0;
  }

  if(kde_arguments.kernel_ == "gaussian") {
    if(kde_arguments.series_expansion_type_ == "hypercube") {
      StartComputation <
      TableType,
      mlpack::series_expansion::GaussianKernelHypercubeAux > (
        kde_arguments);
    }
    else {
      StartComputation <
      TableType,
      mlpack::series_expansion::GaussianKernelMultivariateAux > (
        kde_arguments);
    }
  }
  else {

    // Only the multivariate expansion is available for the
    // Epanechnikov.
    StartComputation <
    TableType,
    mlpack::series_expansion::EpanKernelMultivariateAux > (
      kde_arguments);
  }

  return 0;
}
