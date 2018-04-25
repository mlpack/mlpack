/**
 * @file kde_main.cpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * Executable for running Kernel Density Estimation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/tree/statistic.hpp>

#include "kde.hpp"

using namespace mlpack;
using namespace mlpack::kde;
using namespace mlpack::util;
using namespace std;

// Define parameters for the executable.
PROGRAM_INFO("Kernel Density Estimation", "This program performs a Kernel "
    "Density Estimation for a given reference dataset.");

// Required options.
PARAM_DOUBLE_IN_REQ("bandwidth", "Bandwidth of the kernel", "b");
PARAM_MATRIX_IN_REQ("reference", "Input dataset to KDE on.", "i");
PARAM_MATRIX_IN_REQ("query", "Query dataset to KDE on.", "q");

// Configuration options
PARAM_STRING_IN("kernel", "Kernel to use for the estimation"
    "('gaussian').", "k", "gaussian");
PARAM_STRING_IN("tree", "Tree to use for the estimation"
    "('kd-tree', 'ball-tree).", "t", "kd-tree");
PARAM_STRING_IN("metric", "Metric to use for the estimation"
    "('euclidean').", "m", "euclidean");
PARAM_INT_IN("leaf_size", "Leaf size to use for the tree", "l", 2);
PARAM_DOUBLE_IN("error", "Relative error tolerance for the result" , "e", 1e-8);
PARAM_FLAG("breadth_first", "Use breadth-first traversal instead of depth"
           "first.", "w");

// Output options.
PARAM_MATRIX_OUT("output", "Matrix to store output estimations.",
    "o");

static void mlpackMain()
{
  arma::mat reference = std::move(CLI::GetParam<arma::mat>("reference"));
  arma::mat query = std::move(CLI::GetParam<arma::mat>("query"));
  double error = CLI::GetParam<double>("error");
  double bandwidth = CLI::GetParam<double>("bandwidth");
  int leafSize = CLI::GetParam<int>("leaf_size");

  arma::vec estimations = std::move(arma::vec(query.n_cols, arma::fill::zeros));
  kde::KDE<mlpack::metric::EuclideanDistance,
           arma::mat,
           kernel::GaussianKernel,
           tree::KDTree>
    model = kde::KDE<>(reference, error, bandwidth, leafSize);

  model.Evaluate(query, estimations);
  // Output estimations to file if defined.
  if (CLI::HasParam("output"))
  {
    CLI::GetParam<arma::mat>("output") = std::move(estimations);
  }
  else
  {
    std::cout.precision(40);
    estimations.raw_print(std::cout);
  }
}
