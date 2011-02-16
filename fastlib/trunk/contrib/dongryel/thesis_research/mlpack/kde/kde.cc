/** @file kde.cc
 *
 *  The main driver for the KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <armadillo>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "core/tree/gen_metric_tree.h"
#include "mlpack/kde/kde_dev.h"
#include "mlpack/series_expansion/kernel_aux.h"

template<typename KernelAuxType>
void StartComputation(boost::program_options::variables_map &vm) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree <
  mlpack::kde::KdeStatistic <
  KernelAuxType::ExpansionType > > > TableType;

  // Parse arguments for Kde.
  mlpack::kde::KdeArguments<TableType> kde_arguments;
  if(mlpack::kde::KdeArgumentParser::ParseArguments(vm, &kde_arguments)) {
    return;
  }

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

  boost::program_options::variables_map vm;
  if(mlpack::kde::KdeArgumentParser::ConstructBoostVariableMap(
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
