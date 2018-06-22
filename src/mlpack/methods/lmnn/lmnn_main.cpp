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

#include <mlpack/core/optimizers/bigbatch_sgd/bigbatch_sgd.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
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
    "The program also requires number of targets neighbors to work with ( "
    "specified with " + PRINT_PARAM_STRING("num_targets") + "), A "
    "regularization parameter can also be passed, It acts as a trade of "
    "between the pulling and pushing terms  (specified with " +
    PRINT_PARAM_STRING("regularization") + "), In addition, this "
    "implementation of LMNN includes a parameter to decide the interval "
    "after which impostors must be re-calculated (specified with " +
    PRINT_PARAM_STRING("range") + ")."
    "\n\n"
    "This implementation of LMNN uses AdaGrad, BigBatch_SGD, stochastic "
    "gradient descent, mini-batch stochastic gradient descent, or the L_BFGS "
    "optimizer. "
    "\n\n"
    "AdaGrad, specified by the value 'adagrad' for the parameter " +
    PRINT_PARAM_STRING("optimizer") + ", uses maximum of past squared "
    "gradients. It primarily on six parameters: the step size (specified "
    "with " + PRINT_PARAM_STRING("step_size") + "), the batch size (specified "
    "with " + PRINT_PARAM_STRING("batch_size") + "), the maximum number of "
    "passes (specified with " + PRINT_PARAM_STRING("passes") + ") "
    ", the exponential decay rate for the first moment estimates (specified "
    "with " + PRINT_PARAM_STRING("beta1") + "), the exponential decay rate for "
    "weighted infinity norm (specified with " + PRINT_PARAM_STRING("beta2") +
     "), epsilon (specified with " + PRINT_PARAM_STRING("epsilon") + ").  In "
    "addition, a normalized starting point can be used by specifying the " +
    PRINT_PARAM_STRING("normalize") + " parameter. "
    "\n\n"
    "BigBatch_SGD, specified by the value 'bbsgd' for the parameter " +
    PRINT_PARAM_STRING("optimizer") + ", depends primarily on four parameters: "
    "the step size (specified with " + PRINT_PARAM_STRING("step_size") + "), "
    "the batch size (specified with " + PRINT_PARAM_STRING("batch_size") + "), "
    "the maximum number of passes (specified with " +
    PRINT_PARAM_STRING("passes") + "), the batch delta (specified "
    "with " + PRINT_PARAM_STRING("batch_delta") + ").  In addition, "
    "a normalized starting point can be used by specifying the " +
    PRINT_PARAM_STRING("normalize") + " parameter. "
    "\n\n"
    "Stochastic gradient descent, specified by the value 'sgd' for the "
    "parameter " + PRINT_PARAM_STRING("optimizer") + ", depends "
    "primarily on three parameters: the step size (specified with " +
    PRINT_PARAM_STRING("step_size") + "), the batch size (specified with " +
    PRINT_PARAM_STRING("batch_size") + "), and the maximum number of passes "
    "(specified with " + PRINT_PARAM_STRING("passes") + ").  In "
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
    "literature on L-BFGS.  In addition, a normalized starting point can be "
    "used by specifying the " + PRINT_PARAM_STRING("normalize") + " parameter."
    "\n\n"
    "By default, the AMSGrad optimizer is used."
    "\n\n"
    "Example - Let's say we want to learn distance on iris dataset with "
    "number of targets as 3 using BigBatch_SGD optimizer. A simple call for "
    "the same will look like: "
    "\n\n"
    "$ bin/mlpack_lmnn -i iris.csv -l iris_labels.txt -k 3 -O bbsgd -o "
    "output.csv --verbose"
    "\n\n"
    "An another program call making use of range & regularization parameter "
    "with dataset having labels as last column can be made as: "
    "\n\n"
    "$ bin/mlpack_lmnn -i letter_recognition.csv -k 5 -R 10 -r 0.4 -o "
    "output.csv --verbose");

PARAM_MATRIX_IN_REQ("input", "Input dataset to run LMNN on.", "i");
PARAM_UROW_IN("labels", "Labels for input dataset.", "l");
PARAM_INT_IN("num_targets", "Number of target neighbors to use for each "
    "datapoint.", "k", 1);
PARAM_MATRIX_OUT("output", "Output matrix for learned distance matrix.", "o");
PARAM_STRING_IN("optimizer", "Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or "
    "'lbfgs'.", "O", "amsgrad");
PARAM_DOUBLE_IN("regularization", "Regularization for LMNN objective function ",
    "r", 0.5);
PARAM_FLAG("normalize", "Use a normalized starting point for optimization. It"
    "is useful for when points are far apart, or when SGD is returning NaN.",
    "N");
PARAM_INT_IN("passes", "Maximum number of full passes over dataset for "
    "AMSGrad, BB_SGD and SGD.", "p", 50);
PARAM_INT_IN("max_iterations", "Maximum number of iterations for "
    "L-BFGS (0 indicates no limit).", "n", 100000);
PARAM_DOUBLE_IN("tolerance", "Maximum tolerance for termination of AMSGrad, "
    "BB_SGD, SGD or L-BFGS.", "t", 1e-7);
PARAM_DOUBLE_IN("step_size", "Step size for AMSGrad, BB_SGD and SGD (alpha).",
    "a", 0.01);
PARAM_FLAG("linear_scan", "Don't shuffle the order in which data points are "
    "visited for SGD or mini-batch SGD.", "L");
PARAM_INT_IN("batch_size", "Batch size for mini-batch SGD.", "b", 50);

PARAM_DOUBLE_IN("beta1", "Exponential decay rate for the first moment "
    "estimates of AMSGrad.", "x", 0.9);
PARAM_DOUBLE_IN("beta2", "Exponential decay rate for weighted infinity norm "
    "estimates of AMSGrad.", "y", 0.999);
PARAM_DOUBLE_IN("epsilon", "Value used to initialise the mean squared gradient "
    "parameter of AMSGrad.", "e", 1e-8);
PARAM_DOUBLE_IN("batch_delta", "Factor for the batch update step of "
  "BigBatch_SGD.", "d", 0.1);
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
PARAM_INT_IN("range", "Number of iterations after which impostors needs to be "
    "recalculated", "R", 1);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace mlpack;
using namespace mlpack::lmnn;
using namespace mlpack::metric;
using namespace mlpack::optimization;
using namespace mlpack::util;
using namespace std;

static void mlpackMain()
{
  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");

  const string optimizerType = CLI::GetParam<string>("optimizer");
  RequireParamInSet<string>("optimizer", { "amsgrad", "bbsgd", "sgd",
       "lbfgs" }, true, "unknown optimizer type");

  // Warn on unused parameters.
  if (optimizerType == "amsgrad")
  {
    ReportIgnoredParam("num_basis", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("armijo_constant", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("wolfe", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_line_search_trials",
        "L-BFGS optimizer is not being used");
    ReportIgnoredParam("min_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("num_basis", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("batch_delta",
        "BigBatch_SGD optimizer is not being used");
    ReportIgnoredParam("max_iterations", "L-BFGS optimizer is not being used");
  }
  else if (optimizerType == "bbsgd")
  {
    ReportIgnoredParam("num_basis", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("armijo_constant", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("wolfe", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_line_search_trials",
        "L-BFGS optimizer is not being used");
    ReportIgnoredParam("min_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("num_basis", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("beta1", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("beta2", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("epsilon", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("max_iterations", "L-BFGS optimizer is not being used");
  }
  else if (optimizerType == "sgd")
  {
    ReportIgnoredParam("num_basis", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("armijo_constant", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("wolfe", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_line_search_trials",
        "L-BFGS optimizer is not being used");
    ReportIgnoredParam("min_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("max_step", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("num_basis", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("beta1", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("beta2", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("epsilon", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("max_iterations", "L-BFGS optimizer is not being used");
    ReportIgnoredParam("batch_delta",
        "BigBatch_SGD optimizer is not being used");
  }
  else if (optimizerType == "lbfgs")
  {
    ReportIgnoredParam("step_size", "SGD optimizer is not being used");
    ReportIgnoredParam("linear_scan", "SGD optimizer is not being used");
    ReportIgnoredParam("batch_size", "SGD optimizer is not being used");
    ReportIgnoredParam("beta1", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("beta2", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("epsilon", "AMSGrad optimizer is not being used");
    ReportIgnoredParam("batch_delta",
        "BigBatch_SGD optimizer is not being used");
  }

  const size_t numTargets = (size_t) CLI::GetParam<int>("num_targets");
  const double regularization = CLI::GetParam<double>("regularization");
  const double stepSize = CLI::GetParam<double>("step_size");
  const size_t passes = (size_t) CLI::GetParam<int>("passes");
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
  const double beta1 = CLI::GetParam<double>("beta1");
  const double beta2 = CLI::GetParam<double>("beta2");
  const double epsilon = CLI::GetParam<double>("epsilon");
  const double batchDelta = CLI::GetParam<double>("batch_delta");
  const size_t range = (size_t) CLI::GetParam<int>("range");

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
  if (optimizerType == "amsgrad")
  {
    LMNN<LMetric<2>> lmnn(data, labels, numTargets);
    lmnn.Regularization() = regularization;
    lmnn.Range() = range;
    lmnn.Optimizer().StepSize() = stepSize;
    lmnn.Optimizer().MaxIterations() = passes * data.n_cols;
    lmnn.Optimizer().Beta1() = beta1;
    lmnn.Optimizer().Beta2() = beta2;
    lmnn.Optimizer().Epsilon() = epsilon;
    lmnn.Optimizer().Tolerance() = tolerance;
    lmnn.Optimizer().Shuffle() = shuffle;
    lmnn.Optimizer().BatchSize() = batchSize;

    lmnn.LearnDistance(distance);
  }
  else if (optimizerType == "bbsgd")
  {
    LMNN<LMetric<2>, BBS_BB> lmnn(data, labels, numTargets);
    lmnn.Regularization() = regularization;
    lmnn.Range() = range;
    lmnn.Optimizer().StepSize() = stepSize;
    lmnn.Optimizer().BatchDelta() = batchDelta;
    lmnn.Optimizer().MaxIterations() = passes * data.n_cols;
    lmnn.Optimizer().Tolerance() = tolerance;
    lmnn.Optimizer().Shuffle() = shuffle;
    lmnn.Optimizer().BatchSize() = batchSize;

    lmnn.LearnDistance(distance);
  }
  else if (optimizerType == "sgd")
  {
    // Using SGD is not recommended as the learning matrix can
    // diverge to inf causing serious memory problems.
    LMNN<LMetric<2>, StandardSGD> lmnn(data, labels, numTargets);
    lmnn.Regularization() = regularization;
    lmnn.Range() = range;
    lmnn.Optimizer().StepSize() = stepSize;
    lmnn.Optimizer().MaxIterations() = passes * data.n_cols;
    lmnn.Optimizer().Tolerance() = tolerance;
    lmnn.Optimizer().Shuffle() = shuffle;
    lmnn.Optimizer().BatchSize() = batchSize;

    lmnn.LearnDistance(distance);
  }
  else if (optimizerType == "lbfgs")
  {
    LMNN<LMetric<2>, L_BFGS> lmnn(data, labels, numTargets);
    lmnn.Regularization() = regularization;
    lmnn.Range() = range;
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
    CLI::GetParam<arma::mat>("output") = std::move(distance);
}
