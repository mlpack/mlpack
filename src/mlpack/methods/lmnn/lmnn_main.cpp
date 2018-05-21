/**
 * @file lmnn_main.cpp
 * @author Manish Kumar
 *
 * Executable for Large Margin Nearest Neighbors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/normalize_labels.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "lmnn.hpp"

#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

// Define parameters.
PROGRAM_INFO("Large Margin Nearest Neighbors (LMNN)",
    "This program implements Large Margin Nearest Neighbors, a distance "
    "learning technique.  The method seeks to improve k-nearest-neighbor "
    "classification on a dataset.  The method employes the strategy of "
    "reducing distance between similar labeled data points (a.k.a target "
    "neighbors) and increasing distance between differently labeled points"
    " (a.k.a impostors) using standard optimization techniques over the "
    "gradient of the distance between data points."
    "\n\n"
    "To work, this algorithm needs labeled data.  It can be given as the last "
    "row of the input dataset (specified with " + PRINT_PARAM_STRING("input") +
    "), or alternatively as a separate matrix (specified with " +
    PRINT_PARAM_STRING("labels") + ")."
    "\n\n"
    "This implementation of LMNN uses stochastic gradient descent, mini-batch "
    "stochastic gradient descent, or the L_BFGS optimizer. "
    "\n\n"
    "Stochastic gradient descent, specified by the value 'sgd' for the "
    "parameter " + PRINT_PARAM_STRING("optimizer") + ", depends "
    "primarily on three parameters: the step size (specified with " +
    PRINT_PARAM_STRING("step_size") + "), the batch size (specified with " +
    PRINT_PARAM_STRING("batch_size") + "), and the maximum number of iterations"
    " (specified with " + PRINT_PARAM_STRING("max_iterations") + ").  In "
    "addition, a normalized starting point can be used by specifying the " +
    PRINT_PARAM_STRING("normalize") + " parameter. "
    "\n\n"
    "The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter " +
    PRINT_PARAM_STRING("optimizer") + ", uses a back-tracking line search "
    "algorithm to minimize a function.  The following parameters are used by "
    "L-BFGS: " + PRINT_PARAM_STRING("num_basis") + " (specifies the number"
    " of memory points used by L-BFGS), " +
    PRINT_PARAM_STRING("max_iterations") + ", " +
    PRINT_PARAM_STRING("armijo_constant") + ", " +
    PRINT_PARAM_STRING("wolfe") + ", " + PRINT_PARAM_STRING("tolerance") +
    " (the optimization is terminated when the gradient norm is below this "
    "value), " + PRINT_PARAM_STRING("max_line_search_trials") + ", " +
    PRINT_PARAM_STRING("min_step") + ", and " +
    PRINT_PARAM_STRING("max_step") + " (which both refer to the line search "
    "routine).  For more details on the L-BFGS optimizer, consult either the "
    "mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published "
    "literature on L-BFGS."
    "\n\n"
    "By default, the SGD optimizer is used.");

PARAM_MATRIX_IN_REQ("input", "Input dataset to run LMNN on.", "i");
PARAM_UROW_IN("labels", "Labels for input dataset.", "l");
PARAM_INT_IN("num_targets", "Number of target neighbors to use for each"
    " datapoint.", "k", 1);
PARAM_MATRIX_OUT("output", "Output matrix for learned distance matrix.", "o");
PARAM_STRING_IN("optimizer", "Optimizer to use; 'sgd' or 'lbfgs'.", "O", "sgd");

PARAM_DOUBLE_IN("regularization", "Regularization for LMNN objective function ",
    "r", 0.5);

PARAM_FLAG("normalize", "Use a normalized starting point for optimization. This"
    " is useful for when points are far apart, or when SGD is returning NaN.",
    "N");

PARAM_INT_IN("max_iterations", "Maximum number of iterations for SGD or L-BFGS "
    "(0 indicates no limit).", "n", 100000);
PARAM_DOUBLE_IN("tolerance", "Maximum tolerance for termination of SGD or "
    "L-BFGS.", "t", 1e-7);

PARAM_DOUBLE_IN("step_size", "Step size for stochastic gradient descent "
    "(alpha).", "a", 0.01);
PARAM_FLAG("linear_scan", "Don't shuffle the order in which data points are "
    "visited for SGD or mini-batch SGD.", "L");
PARAM_INT_IN("batch_size", "Batch size for mini-batch SGD.", "b", 50);

PARAM_INT_IN("num_basis", "Number of memory points to be stored for L-BFGS.",
    "B", 5);
PARAM_DOUBLE_IN("armijo_constant", "Armijo constant for L-BFGS.", "A", 1e-4);
PARAM_DOUBLE_IN("wolfe", "Wolfe condition parameter for L-BFGS.", "w", 0.9);
PARAM_INT_IN("max_line_search_trials", "Maximum number of line search trials "
    "for L-BFGS.", "T", 50);
PARAM_DOUBLE_IN("min_step", "Minimum step of line search for L-BFGS.", "m",
    1e-20);
PARAM_DOUBLE_IN("max_step", "Maximum step of line search for L-BFGS.", "M",
    1e20);

using namespace mlpack;
using namespace mlpack::lmnn;
using namespace mlpack::metric;
using namespace mlpack::optimization;
using namespace mlpack::util;
using namespace std;

static void mlpackMain()
{
  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");

  const string optimizerType = CLI::GetParam<string>("optimizer");
  RequireParamInSet<string>("optimizer", { "sgd", "lbfgs" },
      true, "unknown optimizer type");

  // Warn on unused parameters.
  if (optimizerType == "sgd")
  {
    ReportIgnoredParam("num_basis", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("armijo_constant", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("wolfe", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_line_search_trials",
        "L-BFGS optimizer is not being used");
    ReportIgnoredParam("min_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("batch_size", "L-BFGS optimizer is not being used");
  }
  else if (optimizerType == "lbfgs")
  {
    ReportIgnoredParam("step_size", "SGD optimizer is not being used");
    ReportIgnoredParam("linear_scan", "SGD optimizer is not being used");
    ReportIgnoredParam("batch_size", "SGD optimizer is not being used");
  }

  const size_t numTargets = (size_t) CLI::GetParam<int>("num_targets");
  const double regularization = CLI::GetParam<double>("regularization");
  const double stepSize = CLI::GetParam<double>("step_size");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const bool normalize = CLI::HasParam("normalize");
  const bool shuffle = !CLI::HasParam("linear_scan");
  const int numBasis = CLI::GetParam<int>("num_basis");
  const double armijoConstant = CLI::GetParam<double>("armijo_constant");
  const double wolfe = CLI::GetParam<double>("wolfe");
  const int maxLineSearchTrials = CLI::GetParam<int>("max_line_search_trials");
  const double minStep = CLI::GetParam<double>("min_step");
  const double maxStep = CLI::GetParam<double>("max_step");
  const size_t batchSize = (size_t) CLI::GetParam<int>("batch_size");

  // Load data.
  arma::mat data = std::move(CLI::GetParam<arma::mat>("input"));

  // Do we want to load labels separately?
  arma::Row<size_t> rawLabels(data.n_cols);
  if (CLI::HasParam("labels"))
  {
    rawLabels = std::move(CLI::GetParam<arma::Row<size_t>>("labels"));
  }
  else
  {
    Log::Info << "Using last column of input dataset as labels." << endl;
    for (size_t i = 0; i < data.n_cols; i++)
      rawLabels[i] = (size_t) data(data.n_rows - 1, i);

    data.shed_row(data.n_rows - 1);
  }

  // Now, normalize the labels.
  arma::Col<size_t> mappings;
  arma::Row<size_t> labels;
  data::NormalizeLabels(rawLabels, labels, mappings);

  arma::mat distance;

  // Normalize the data, if necessary.
  if (normalize)
  {
    // Find the minimum and maximum values for each dimension.
    arma::vec ranges = arma::max(data, 1) - arma::min(data, 1);
    for (size_t d = 0; d < ranges.n_elem; ++d)
      if (ranges[d] == 0.0)
        ranges[d] = 1; // A range of 0 produces NaN later on.

    distance = diagmat(1.0 / ranges);
    Log::Info << "Using normalized starting point for optimization."
        << endl;
  }
  else
  {
    distance.eye();
  }

  // Now create the LMNN object and run the optimization.
  if (optimizerType == "sgd")
  {
    LMNN<LMetric<2> > lmnn(data, labels, numTargets);
    lmnn.Regularization() = regularization;
    lmnn.Optimizer().StepSize() = stepSize;
    lmnn.Optimizer().MaxIterations() = maxIterations;
    lmnn.Optimizer().Tolerance() = tolerance;
    lmnn.Optimizer().Shuffle() = shuffle;
    lmnn.Optimizer().BatchSize() = batchSize;

    lmnn.LearnDistance(distance);
  }
  else if (optimizerType == "lbfgs")
  {
    LMNN<LMetric<2>, L_BFGS> lmnn(data, labels, numTargets);
    lmnn.Regularization() = regularization;
    lmnn.Optimizer().NumBasis() = numBasis;
    lmnn.Optimizer().MaxIterations() = maxIterations;
    lmnn.Optimizer().ArmijoConstant() = armijoConstant;
    lmnn.Optimizer().Wolfe() = wolfe;
    lmnn.Optimizer().MinGradientNorm() = tolerance;
    lmnn.Optimizer().MaxLineSearchTrials() = maxLineSearchTrials;
    lmnn.Optimizer().MinStep() = minStep;
    lmnn.Optimizer().MaxStep() = maxStep;

    lmnn.LearnDistance(distance);
  }

  // Save the output.
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(distance * data);
}
