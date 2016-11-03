/**
 * @file nca_main.cpp
 * @author Ryan Curtin
 *
 * Executable for Neighborhood Components Analysis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include "nca.hpp"

#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>

// Define parameters.
PROGRAM_INFO("Neighborhood Components Analysis (NCA)",
    "This program implements Neighborhood Components Analysis, both a linear "
    "dimensionality reduction technique and a distance learning technique.  The"
    " method seeks to improve k-nearest-neighbor classification on a dataset "
    "by scaling the dimensions.  The method is nonparametric, and does not "
    "require a value of k.  It works by using stochastic (\"soft\") neighbor "
    "assignments and using optimization techniques over the gradient of the "
    "accuracy of the neighbor assignments."
    "\n\n"
    "To work, this algorithm needs labeled data.  It can be given as the last "
    "row of the input dataset (--input_file), or alternatively in a separate "
    "file (--labels_file)."
    "\n\n"
    "This implementation of NCA uses stochastic gradient descent, mini-batch "
    "stochastic gradient descent, or the L_BFGS optimizer.  These optimizers do"
    " not guarantee global convergence for a nonconvex objective function "
    "(NCA's objective function is nonconvex), so the final results could depend"
    " on the random seed or other optimizer parameters."
    "\n\n"
    "Stochastic gradient descent, specified by --optimizer \"sgd\", depends "
    "primarily on two parameters: the step size (--step_size) and the maximum "
    "number of iterations (--max_iterations).  In addition, a normalized "
    "starting point can be used (--normalize), which is necessary if many "
    "warnings of the form 'Denominator of p_i is 0!' are given.  Tuning the "
    "step size can be a tedious affair.  In general, the step size is too large"
    " if the objective is not mostly uniformly decreasing, or if zero-valued "
    "denominator warnings are being issued.  The step size is too small if the "
    "objective is changing very slowly.  Setting the termination condition can "
    "be done easily once a good step size parameter is found; either increase "
    "the maximum iterations to a large number and allow SGD to find a minimum, "
    "or set the maximum iterations to 0 (allowing infinite iterations) and set "
    "the tolerance (--tolerance) to define the maximum allowed difference "
    "between objectives for SGD to terminate.  Be careful---setting the "
    "tolerance instead of the maximum iterations can take a very long time and "
    "may actually never converge due to the properties of the SGD optimizer. "
    "Note that a single iteration of SGD refers to a single point, so to take "
    "a single pass over the dataset, set --max_iterations equal to the number "
    "of points in the dataset."
    "\n\n"
    "The mini-batch SGD optimizer, specified by --optimizer \"minibatch-sgd\", "
    "has the same parameters as SGD, but the batch size may also be specified "
    "with the --batch_size (-b) option.  Each iteration of mini-batch SGD "
    "refers to a single mini-batch."
    "\n\n"
    "The L-BFGS optimizer, specified by --optimizer \"lbfgs\", uses a "
    "back-tracking line search algorithm to minimize a function.  The "
    "following parameters are used by L-BFGS: --num_basis (specifies the number"
    " of memory points used by L-BFGS), --max_iterations, --armijo_constant, "
    "--wolfe, --tolerance (the optimization is terminated when the gradient "
    "norm is below this value), --max_line_search_trials, --min_step and "
    "--max_step (which both refer to the line search routine).  For more "
    "details on the L-BFGS optimizer, consult either the mlpack L-BFGS "
    "documentation (in lbfgs.hpp) or the vast set of published literature on "
    "L-BFGS."
    "\n\n"
    "By default, the SGD optimizer is used.");

PARAM_MATRIX_IN_REQ("input", "Input dataset to run NCA on.", "i");
PARAM_MATRIX_OUT("output", "Output matrix for learned distance matrix.", "o");
PARAM_UMATRIX_IN("labels", "Labels for input dataset.", "l");
PARAM_STRING_IN("optimizer", "Optimizer to use; 'sgd', 'minibatch-sgd', or "
    "'lbfgs'.", "O", "sgd");

PARAM_FLAG("normalize", "Use a normalized starting point for optimization. This"
    " is useful for when points are far apart, or when SGD is returning NaN.",
    "N");

PARAM_INT_IN("max_iterations", "Maximum number of iterations for SGD or L-BFGS "
    "(0 indicates no limit).", "n", 500000);
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

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace mlpack;
using namespace mlpack::nca;
using namespace mlpack::metric;
using namespace mlpack::optimization;
using namespace std;

int main(int argc, char* argv[])
{
  // Parse command line.
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  if (!CLI::HasParam("output"))
    Log::Warn << "--output_file (-o) not specified; no output will be saved!"
        << endl;

  const string optimizerType = CLI::GetParam<string>("optimizer");

  if ((optimizerType != "sgd") && (optimizerType != "lbfgs") &&
      (optimizerType != "minibatch-sgd"))
  {
    Log::Fatal << "Optimizer type '" << optimizerType << "' unknown; must be "
        << "'sgd', 'minibatch-sgd', or 'lbfgs'!" << endl;
  }

  // Warn on unused parameters.
  if (optimizerType == "sgd" || optimizerType == "minibatch-sgd")
  {
    if (CLI::HasParam("num_basis"))
      Log::Warn << "Parameter --num_basis ignored (not using 'lbfgs' "
          << "optimizer)." << endl;

    if (CLI::HasParam("armijo_constant"))
      Log::Warn << "Parameter --armijo_constant ignored (not using 'lbfgs' "
          << "optimizer)." << endl;

    if (CLI::HasParam("wolfe"))
      Log::Warn << "Parameter --wolfe ignored (not using 'lbfgs' optimizer).\n";

    if (CLI::HasParam("max_line_search_trials"))
      Log::Warn << "Parameter --max_line_search_trials ignored (not using "
          << "'lbfgs' optimizer." << endl;

    if (CLI::HasParam("min_step"))
      Log::Warn << "Parameter --min_step ignored (not using 'lbfgs' optimizer)."
          << endl;

    if (CLI::HasParam("max_step"))
      Log::Warn << "Parameter --max_step ignored (not using 'lbfgs' optimizer)."
          << endl;

    if (optimizerType == "sgd" && CLI::HasParam("batch_size"))
      Log::Warn << "Parameter --batch_size ignored (not using 'minibatch-sgd' "
          << "optimizer." << endl;
  }
  else if (optimizerType == "lbfgs")
  {
    if (CLI::HasParam("step_size"))
      Log::Warn << "Parameter --step_size ignored (not using 'sgd' or "
          << "'minibatch-sgd' optimizer)." << endl;

    if (CLI::HasParam("linear_scan"))
      Log::Warn << "Parameter --linear_scan ignored (not using 'sgd' or "
          << "'minibatch-sgd' optimizer)." << endl;

    if (CLI::HasParam("batch_size"))
      Log::Warn << "Parameter --batch_size ignored (not using 'minibatch-sgd' "
          << "optimizer)." << endl;
  }

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
  arma::Mat<size_t> rawLabels(1, data.n_cols);
  if (CLI::HasParam("labels"))
  {
    rawLabels = std::move(CLI::GetParam<arma::Mat<size_t>>("labels"));

    if (rawLabels.n_cols == 1)
      rawLabels = trans(rawLabels);

    if (rawLabels.n_rows > 1)
      Log::Fatal << "Labels must have only one column or row!" << endl;
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
  data::NormalizeLabels(rawLabels.row(0), labels, mappings);

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

  // Now create the NCA object and run the optimization.
  if (optimizerType == "sgd")
  {
    NCA<LMetric<2> > nca(data, labels);
    nca.Optimizer().StepSize() = stepSize;
    nca.Optimizer().MaxIterations() = maxIterations;
    nca.Optimizer().Tolerance() = tolerance;
    nca.Optimizer().Shuffle() = shuffle;

    nca.LearnDistance(distance);
  }
  else if (optimizerType == "lbfgs")
  {
    NCA<LMetric<2>, L_BFGS> nca(data, labels);
    nca.Optimizer().NumBasis() = numBasis;
    nca.Optimizer().MaxIterations() = maxIterations;
    nca.Optimizer().ArmijoConstant() = armijoConstant;
    nca.Optimizer().Wolfe() = wolfe;
    nca.Optimizer().MinGradientNorm() = tolerance;
    nca.Optimizer().MaxLineSearchTrials() = maxLineSearchTrials;
    nca.Optimizer().MinStep() = minStep;
    nca.Optimizer().MaxStep() = maxStep;

    nca.LearnDistance(distance);
  }
  else if (optimizerType == "minibatch-sgd")
  {
    NCA<LMetric<2>, MiniBatchSGD> nca(data, labels);
    nca.Optimizer().StepSize() = stepSize;
    nca.Optimizer().MaxIterations() = maxIterations;
    nca.Optimizer().Tolerance() = tolerance;
    nca.Optimizer().Shuffle() = shuffle;
    nca.Optimizer().BatchSize() = batchSize;

    nca.LearnDistance(distance);
  }

  // Save the output.
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(distance);
}
