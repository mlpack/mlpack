/** @file kde.cc
 *
 *  The main driver for the KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <iostream>
#include <string>
#include <armadillo>
#include "kde_dev.h"
#include "core/tree/gen_metric_tree.h"

int main(int argc, char *argv[]) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::table::DensePoint> > TableType;

  // Parse arguments for Kde.
  ml::KdeArguments<TableType> kde_arguments;
  ml::Kde<TableType>::ParseArguments(argc, argv, &kde_arguments);

  // Instantiate a KDE object.
  ml::Kde<TableType> kde_instance;
  kde_instance.Init(kde_arguments);

  // Compute the result.
  ml::KdeResult< std::vector<double> > kde_result;
  kde_instance.Compute(kde_arguments, &kde_result);

  // Output the KDE result to the file.
  std::cerr << "Writing the densities to the file: " <<
            kde_arguments.densities_out_ << "\n";
  kde_result.PrintDebug(kde_arguments.densities_out_);

  return 0;
}
