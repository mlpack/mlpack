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
PARAM_MATRIX_IN_REQ("reference", "Input dataset to KDE on.", "r");
PARAM_MATRIX_IN_REQ("query", "Query dataset to KDE on.", "q");
PARAM_DOUBLE_IN_REQ("bandwidth", "Bandwidth of the kernel", "b");

// Configuration options
PARAM_STRING_IN("kernel", "Kernel to use for the estimation"
    "('gaussian', 'epanechnikov').", "k", "gaussian");
PARAM_STRING_IN("tree", "Tree to use for the estimation"
    "('kd-tree', 'ball-tree').", "t", "kd-tree");
PARAM_STRING_IN("metric", "Metric to use for the estimation"
    "('euclidean').", "m", "euclidean");
PARAM_DOUBLE_IN("rel_error",
                "Relative error tolerance for the result",
                "e",
                1e-8);
PARAM_DOUBLE_IN("abs_error",
                "Relative error tolerance for the result",
                "E",
                0.0);
PARAM_FLAG("breadth_first", "Use breadth-first traversal instead of depth"
           "first.", "w");

// Output options.
PARAM_MATRIX_OUT("output", "Matrix to store output estimations.",
    "o");

static void mlpackMain()
{
  // Get all parameters.
  arma::mat reference = std::move(CLI::GetParam<arma::mat>("reference"));
  arma::mat query = std::move(CLI::GetParam<arma::mat>("query"));
  const double bandwidth = CLI::GetParam<double>("bandwidth");
  const std::string kernelStr = CLI::GetParam<std::string>("kernel");
  const std::string treeStr = CLI::GetParam<std::string>("tree");
  const std::string metricStr = CLI::GetParam<std::string>("metric");
  const double relError = CLI::GetParam<double>("rel_error");
  const double absError = CLI::GetParam<double>("abs_error");
  const bool breadthFirst = CLI::GetParam<bool>("breadth_first");
  // Initialize results vector.
  arma::vec estimations = std::move(arma::vec(query.n_cols, arma::fill::zeros));

  // Handle KD-Tree, Gaussian, Euclidean KDE.
  if (treeStr == "kd-tree" &&
      kernelStr == "gaussian" &&
      metricStr == "euclidean")
  {
    kernel::GaussianKernel kernel(bandwidth);
    metric::EuclideanDistance metric;
    kde::KDE<metric::EuclideanDistance,
             arma::mat,
             kernel::GaussianKernel,
             tree::KDTree>
      model(metric, kernel, relError, absError, breadthFirst);
    model.Train(reference);
    model.Evaluate(query, estimations);
    estimations = estimations / (kernel.Normalizer(query.n_rows));
  }

  // Handle Ball-Tree, Gaussian, Euclidean KDE.
  else if (treeStr == "ball-tree" &&
           kernelStr == "gaussian" &&
           metricStr == "euclidean")
  {
    kernel::GaussianKernel kernel(bandwidth);
    metric::EuclideanDistance metric;
    kde::KDE<metric::EuclideanDistance,
             arma::mat,
             kernel::GaussianKernel,
             tree::BallTree>
      model(metric, kernel, relError, absError, breadthFirst);
    model.Train(reference);
    model.Evaluate(query, estimations);
    estimations = estimations / (kernel.Normalizer(query.n_rows));
  }

  // Handle KD-Tree, Epanechnikov, Euclidean KDE.
  else if (treeStr == "kd-tree" &&
           kernelStr == "epanechnikov" &&
           metricStr == "euclidean")
  {
    kernel::EpanechnikovKernel kernel(bandwidth);
    metric::EuclideanDistance metric;
    kde::KDE<metric::EuclideanDistance,
             arma::mat,
             kernel::EpanechnikovKernel,
             tree::KDTree>
      model(metric, kernel, relError, absError, breadthFirst);
    model.Train(reference);
    model.Evaluate(query, estimations);
    estimations = estimations / (kernel.Normalizer(query.n_rows));
  }

  // Handle Ball-Tree, Epanechnikov, Euclidean KDE.
  else if (treeStr == "ball-tree" &&
           kernelStr == "epanechnikov" &&
           metricStr == "euclidean")
  {
    kernel::EpanechnikovKernel kernel(bandwidth);
    metric::EuclideanDistance metric;
    kde::KDE<metric::EuclideanDistance,
             arma::mat,
             kernel::EpanechnikovKernel,
             tree::BallTree>
      model(metric, kernel, relError, absError, breadthFirst);
    model.Train(reference);
    model.Evaluate(query, estimations);
    estimations = estimations / (kernel.Normalizer(query.n_rows));
  }

  // Input parameters are wrong or are not supported yet.
  else
  {
    Log::Fatal << "Input parameters are not valid or are not supported yet."
               << std::endl;
  }
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
