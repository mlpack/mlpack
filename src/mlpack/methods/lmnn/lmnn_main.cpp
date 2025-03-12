/**
 * @file methods/lmnn/lmnn_main.cpp
 * @author Manish Kumar
 *
 * Executable for Large Margin Nearest Neighbors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME lmnn

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "lmnn.hpp"

// Program Name.
BINDING_USER_NAME("Large Margin Nearest Neighbors (LMNN)");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Large Margin Nearest Neighbors (LMNN), a distance "
    "learning technique.  Given a labeled dataset, this learns a transformation"
    " of the data that improves k-nearest-neighbor performance; this can be "
    "useful as a preprocessing step.");

// Long description.
BINDING_LONG_DESC(
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
    PRINT_PARAM_STRING("labels") + ").  Additionally, a starting point for "
    "optimization (specified with " + PRINT_PARAM_STRING("distance") +
    "can be given, having (r x d) dimensionality.  Here r should satisfy "
    "1 <= r <= d, Consequently a Low-Rank matrix will be optimized. "
    "Alternatively, Low-Rank distance can be learned by specifying the " +
    PRINT_PARAM_STRING("rank") + "parameter (A Low-Rank matrix with uniformly "
    "distributed values will be used as initial learning point). "
    "\n\n"
    "The program also requires number of targets neighbors to work with ( "
    "specified with " + PRINT_PARAM_STRING("k") + "), A "
    "regularization parameter can also be passed, It acts as a trade of "
    "between the pulling and pushing terms (specified with " +
    PRINT_PARAM_STRING("regularization") + "), In addition, this "
    "implementation of LMNN includes a parameter to decide the interval "
    "after which impostors must be re-calculated (specified with " +
    PRINT_PARAM_STRING("update_interval") + ")."
    "\n\n"
    "Output can either be the learned distance matrix (specified with " +
    PRINT_PARAM_STRING("output") +"), or the transformed dataset "
    " (specified with " + PRINT_PARAM_STRING("transformed_data") + "), or "
    "both. Additionally mean-centered dataset (specified with " +
    PRINT_PARAM_STRING("centered_data") + ") can be accessed given "
    "mean-centering (specified with " + PRINT_PARAM_STRING("center") +
    ") is performed on the dataset. Accuracy on initial dataset and final "
    "transformed dataset can be printed by specifying the " +
    PRINT_PARAM_STRING("print_accuracy") + "parameter. "
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
    "passes (specified with " + PRINT_PARAM_STRING("passes") + "). In"
    "addition, a normalized starting point can be used by specifying the " +
    PRINT_PARAM_STRING("normalize") + " parameter. "
    "\n\n"
    "BigBatch_SGD, specified by the value 'bbsgd' for the parameter " +
    PRINT_PARAM_STRING("optimizer") + ", depends primarily on four parameters: "
    "the step size (specified with " + PRINT_PARAM_STRING("step_size") + "), "
    "the batch size (specified with " + PRINT_PARAM_STRING("batch_size") + "), "
    "the maximum number of passes (specified with " +
    PRINT_PARAM_STRING("passes") + ").  In addition, a normalized starting "
    "point can be used by specifying the " +
    PRINT_PARAM_STRING("normalize") + " parameter. "
    "\n\n"
    "Stochastic gradient descent, specified by the value 'sgd' for the "
    "parameter " + PRINT_PARAM_STRING("optimizer") + ", depends "
    "primarily on three parameters: the step size (specified with " +
    PRINT_PARAM_STRING("step_size") + "), the batch size (specified with " +
    PRINT_PARAM_STRING("batch_size") + "), and the maximum number of passes "
    "(specified with " + PRINT_PARAM_STRING("passes") + ").  In "
    "addition, a normalized starting point can be used by specifying the " +
    PRINT_PARAM_STRING("normalize") + " parameter. Furthermore, " +
    "mean-centering can be performed on the dataset by specifying the " +
    PRINT_PARAM_STRING("center") + "parameter. "
    "\n\n"
    "The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter " +
    PRINT_PARAM_STRING("optimizer") + ", uses a back-tracking line search "
    "algorithm to minimize a function.  The following parameters are used by "
    "L-BFGS: " + PRINT_PARAM_STRING("max_iterations") + ", " +
    PRINT_PARAM_STRING("tolerance") +
    "(the optimization is terminated when the gradient norm is below this "
    "value).  For more details on the L-BFGS optimizer, consult either the "
    "mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published "
    "literature on L-BFGS.  In addition, a normalized starting point can be "
    "used by specifying the " + PRINT_PARAM_STRING("normalize") + " parameter."
    "\n\n"
    "By default, the AMSGrad optimizer is used.");

// Example.
BINDING_EXAMPLE(
    "Example - Let's say we want to learn distance on iris dataset with "
    "number of targets as 3 using BigBatch_SGD optimizer. A simple call for "
    "the same will look like: "
    "\n\n" +
    PRINT_CALL("lmnn", "input", "iris", "labels", "iris_labels", "k", 3,
    "optimizer", "bbsgd", "output", "output") +
    "\n\n"
    "Another program call making use of update interval & regularization "
    "parameter with dataset having labels as last column can be made as: "
    "\n\n" +
    PRINT_CALL("lmnn", "input", "letter_recognition", "k", 5,
    "update_interval", 10, "regularization", 0.4, "output", "output"));

// See also...
BINDING_SEE_ALSO("@nca", "#nca");
BINDING_SEE_ALSO("Large margin nearest neighbor on Wikipedia",
    "https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor");
BINDING_SEE_ALSO("Distance metric learning for large margin nearest neighbor "
    "classification (pdf)", "https://proceedings.neurips.cc/paper_files/"
    "paper/2005/file/a7f592cef8b130a6967a90617db5681b-Paper.pdf");
BINDING_SEE_ALSO("LMNN C++ class documentation", "@doc/user/methods/lmnn.md");

PARAM_MATRIX_IN_REQ("input", "Input dataset to run LMNN on.", "i");
PARAM_MATRIX_IN("distance", "Initial distance matrix to be used as "
    "starting point", "d");
PARAM_UROW_IN("labels", "Labels for input dataset.", "l");
PARAM_INT_IN("k", "Number of target neighbors to use for each "
    "datapoint.", "k", 1);
PARAM_MATRIX_OUT("output", "Output matrix for learned distance matrix.", "o");
PARAM_MATRIX_OUT("transformed_data", "Output matrix for transformed dataset.",
    "D");
PARAM_MATRIX_OUT("centered_data", "Output matrix for mean-centered dataset.",
    "c");
PARAM_FLAG("print_accuracy", "Print accuracies on initial and transformed "
    "dataset", "P");
PARAM_STRING_IN("optimizer", "Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or "
    "'lbfgs'.", "O", "amsgrad");
PARAM_DOUBLE_IN("regularization", "Regularization for LMNN objective function ",
    "r", 0.5);
PARAM_INT_IN("rank", "Rank of distance matrix to be optimized. ", "A", 0);
PARAM_FLAG("normalize", "Use a normalized starting point for optimization. It"
    "is useful for when points are far apart, or when SGD is returning NaN.",
    "N");
PARAM_FLAG("center", "Perform mean-centering on the dataset. It is useful "
    "when the centroid of the data is far from the origin.", "C");
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
PARAM_INT_IN("update_interval", "Number of iterations after which impostors "
    "need to be recalculated.", "R", 1);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Function to calculate KNN accuracy.
double KNNAccuracy(const arma::mat& dataset,
                   const arma::Row<size_t>& labels,
                   const size_t k)
{
  // Get unique labels.
  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // KNN instance.
  KNN knn;

  knn.Train(dataset);
  knn.Search(k, neighbors, distances);

  // Keep count.
  size_t count = 0;

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    arma::vec Map;
    Map.zeros(uniqueLabels.n_cols);

    for (size_t j = 0; j < k; ++j)
    {
      Map(labels(neighbors(j, i))) +=
          1 / std::pow(distances(j, i) + 1, 2);
    }

    arma::vec index = ConvTo<arma::vec>::From(arma::find(Map == max(Map)));

    // Increase count if labels match.
    if (index(0) == labels(i))
        count++;
  }

  // return accuracy.
  return ((double) count / dataset.n_cols) * 100;
}

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  const string optimizerType = params.Get<string>("optimizer");
  RequireParamInSet<string>(params, "optimizer", { "amsgrad", "bbsgd", "sgd",
       "lbfgs" }, true, "unknown optimizer type");

  // Warn on unused parameters.
  if (optimizerType == "amsgrad")
  {
    ReportIgnoredParam(params, "max_iterations",
        "L-BFGS optimizer is not being used");
  }
  else if (optimizerType == "bbsgd")
  {
    ReportIgnoredParam(params, "max_iterations",
        "L-BFGS optimizer is not being used");
  }
  else if (optimizerType == "sgd")
  {
    ReportIgnoredParam(params, "max_iterations",
        "L-BFGS optimizer is not being used");
  }
  else if (optimizerType == "lbfgs")
  {
    ReportIgnoredParam(params, "step_size", "SGD optimizer is not being used");
    ReportIgnoredParam(params, "linear_scan",
        "SGD optimizer is not being used");
    ReportIgnoredParam(params, "batch_size", "SGD optimizer is not being used");
  }

  RequireParamValue<int>(params, "k", [](int x) { return x > 0; }, true,
      "number of targets must be positive");
  RequireParamValue<int>(params, "update_interval", [](int x) { return x > 0; },
      true, "update interval must be positive");
  RequireParamValue<int>(params, "batch_size", [](int x) { return x > 0; },
      true, "batch size must be positive");
  RequireParamValue<double>(params, "regularization", [](double x)
      { return x >= 0.0; }, true, "regularization value must be non-negative");
  RequireParamValue<double>(params, "step_size", [](double x)
      { return x >= 0.0; }, true, "step size value must be non-negative");
  RequireParamValue<int>(params, "max_iterations", [](int x)
      { return x >= 0; }, true,
      "maximum number of iterations must be non-negative");
  RequireParamValue<int>(params, "passes", [](int x) { return x >= 0; }, true,
      "maximum number of passes must be non-negative");
  RequireParamValue<double>(params, "tolerance",
      [](double x) { return x >= 0.0; }, true,
      "tolerance must be non-negative");
  RequireParamValue<int>(params, "rank", [](int x)
      { return x >= 0; }, true, "rank must be nonnegative");

  const size_t k = (size_t) params.Get<int>("k");
  const double regularization = params.Get<double>("regularization");
  const double stepSize = params.Get<double>("step_size");
  const size_t passes = (size_t) params.Get<int>("passes");
  const size_t maxIterations = (size_t) params.Get<int>("max_iterations");
  const double tolerance = params.Get<double>("tolerance");
  const bool normalize = params.Has("normalize");
  const bool center = params.Has("center");
  const bool printAccuracy = params.Has("print_accuracy");
  const bool shuffle = !params.Has("linear_scan");
  const size_t batchSize = (size_t) params.Get<int>("batch_size");
  const size_t updateInterval = (size_t) params.Get<int>("update_interval");
  const size_t rank = (size_t) params.Get<int>("rank");

  // Load data.
  arma::mat data = std::move(params.Get<arma::mat>("input"));

  // Carry out mean-centering on the dataset, if necessary.
  if (center)
  {
    for (size_t i = 0; i < data.n_rows; ++i)
    {
      data.row(i) -= arma::mean(data.row(i));
    }
  }

  // Do we want to load labels separately?
  arma::Row<size_t> rawLabels(data.n_cols);
  if (params.Has("labels"))
  {
    rawLabels = std::move(params.Get<arma::Row<size_t>>("labels"));
  }
  else
  {
    Log::Info << "Using last column of input dataset as labels." << endl;
    for (size_t i = 0; i < data.n_cols; ++i)
      rawLabels[i] = (size_t) data(data.n_rows - 1, i);

    data.shed_row(data.n_rows - 1);
  }

  // Now, normalize the labels.
  arma::Col<size_t> mappings;
  arma::Row<size_t> labels;
  data::NormalizeLabels(rawLabels, labels, mappings);

  arma::mat distance;

  if (params.Has("distance"))
  {
    distance = std::move(params.Get<arma::mat>("distance"));
  }
  else if (rank)
  {
    distance = arma::randu(rank, data.n_rows);
  }
  // Normalize the data, if necessary.
  else if (normalize)
  {
    // Find the minimum and maximum values for each dimension.
    arma::vec ranges = max(data, 1) - min(data, 1);
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
  timers.Start("lmnn_optimization");
  LMNN lmnn(k, regularization, updateInterval);
  if (optimizerType == "amsgrad")
  {
    ens::AMSGrad opt;
    opt.StepSize() = stepSize;
    opt.MaxIterations() = passes * data.n_cols;
    opt.Tolerance() = tolerance;
    opt.Shuffle() = shuffle;
    opt.BatchSize() = batchSize;

    lmnn.LearnDistance(data, labels, distance, opt);
  }
  else if (optimizerType == "bbsgd")
  {
    ens::BBS_BB opt;
    opt.StepSize() = stepSize;
    opt.MaxIterations() = passes * data.n_cols;
    opt.Tolerance() = tolerance;
    opt.Shuffle() = shuffle;
    opt.BatchSize() = batchSize;

    lmnn.LearnDistance(data, labels, distance, opt);
  }
  else if (optimizerType == "sgd")
  {
    // Using SGD is not recommended as the learning matrix can
    // diverge to inf causing serious memory problems.
    ens::StandardSGD opt;
    opt.StepSize() = stepSize;
    opt.MaxIterations() = passes * data.n_cols;
    opt.Tolerance() = tolerance;
    opt.Shuffle() = shuffle;
    opt.BatchSize() = batchSize;

    lmnn.LearnDistance(data, labels, distance, opt);
  }
  else if (optimizerType == "lbfgs")
  {
    ens::L_BFGS opt;
    opt.MaxIterations() = maxIterations;
    opt.MinGradientNorm() = tolerance;

    lmnn.LearnDistance(data, labels, distance, opt);
  }
  timers.Stop("lmnn_optimization");

  // Print initial & final accuracies if required.
  if (printAccuracy)
  {
    double initAccuracy = KNNAccuracy(data, labels, k);
    double finalAccuracy = KNNAccuracy(distance * data, labels, k);

    Log::Info << "Accuracy on initial dataset: " << initAccuracy <<
        "%" << endl;
    Log::Info << "Accuracy on transformed dataset: " << finalAccuracy <<
        "%"<< endl;
  }

  // Save the output.
  if (params.Has("output"))
    params.Get<arma::mat>("output") = distance;
  if (params.Has("transformed_data"))
    params.Get<arma::mat>("transformed_data") = distance * data;
  if (params.Has("centered_data"))
  {
    if (center)
      params.Get<arma::mat>("centered_data") = std::move(data);
    else
      Log::Info << "Mean-centering was not performed. Centered dataset "
          "will not be saved." << endl;
  }
}
