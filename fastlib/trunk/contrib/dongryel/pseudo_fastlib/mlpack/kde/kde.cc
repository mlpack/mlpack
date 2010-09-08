#include <iostream>
#include <string>
#include <armadillo>
#include "../gnp/dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "kde_dev.h"

int main(int argc, char *argv[]) {

  // Parse arguments for Kde.
  ml::KdeArguments kde_arguments;
  ml::Kde::ParseArguments(argc, argv, &kde_arguments);

  // Instantiate a KDE object.
  ml::DualtreeDfs<ml::Kde> dualtree_dfs;
  ml::Kde kde_instance;
  kde_instance.Init(kde_arguments);

  // Instantiate a dual-tree algorithm of the KDE.
  dualtree_dfs.Init(kde_instance);

  // Compute the result.
  core::metric_kernels::LMetric<2> metric;
  ml::KdeResult< std::vector<double> > kde_result;
  dualtree_dfs.Compute(metric, &kde_result);

  return 0;
}
