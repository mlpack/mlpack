/**
 * @file boostmetric_main.cpp
 * @author Manish Kumar
 *
 * Executable for BoostMetric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/normalize_labels.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>

#include "boostmetric.hpp"

// Define parameters.
PROGRAM_INFO("BoostMetric Distance Learning",
    "This program implements BoostMetric, a distance metric learning "
    "technique.  The method seeks to improve k-nearest-neighbor "
    "classification on a dataset.  The method employes the strategy of "
    "reducing distance between similar labeled data points (a.k.a target "
    "neighbors) and increasing distance between differently labeled points "
    "(a.k.a impostors) using an adaptation of coordinate descent optimization "
    "as a boosting technique."
    "\n\n"
    "To work, this algorithm needs labeled data.  It can be given as the last "
    "row of the input dataset (specified with " + PRINT_PARAM_STRING("input") +
    "), or alternatively as a separate matrix (specified with " +
    PRINT_PARAM_STRING("labels") + ").  Additionally, a starting point for "
    "optimization (specified with " + PRINT_PARAM_STRING("distance") +
    "can be given, having (d x d) dimensionality. "
    "\n\n"
    "Output can be either the learned distance matrix (specified with " +
    PRINT_PARAM_STRING("output") +"), or the transformed dataset "
    "(specified with " + PRINT_PARAM_STRING("transformed_data") + "), or "
    "both.  Accuracy on initial dataset and final transformed dataset can "
    "be printed by specifying the " + PRINT_PARAM_STRING("print_accuracy") +
    "parameter. "
    "\n\n"
    "The algorithm uses binary search for approximating the weight update, "
    "which requires following parameters : " +
    PRINT_PARAM_STRING("wTolerance") + ", " + PRINT_PARAM_STRING("wHigh") +
    " and " + PRINT_PARAM_STRING("wLow") + "."
    "\n\n"
    "Example - Let's say we want to learn distance on iris dataset with "
    "number of targets as 3. A  straightforward call for the same will"
    "look like: "
    "\n\n" +
    PRINT_CALL("mlpack_boostmetric", "input", "iris", "labels", "iris_labels",
    "k", 3, "output", "output") +
    "\n\n"
    "An another program call making use of tolerance with dataset having "
    "labels as last column can be made as: "
    "\n\n" +
    PRINT_CALL("mlpack_boostmetric", "input", "letter_recognition", "k", 5,
    "tolerance", 1e-6, "output", "output"));

PARAM_MATRIX_IN_REQ("input", "Input dataset to run LMNN on.", "i");
PARAM_MATRIX_IN("distance", "Initial distance matrix to be used as "
    "starting point", "d");
PARAM_UROW_IN("labels", "Labels for input dataset.", "l");
PARAM_INT_IN("k", "Number of target neighbors to use for each "
    "datapoint.", "k", 1);
PARAM_MATRIX_OUT("output", "Output matrix for learned distance matrix.", "o");
PARAM_MATRIX_OUT("transformed_data", "Output matrix for transformed dataset.",
    "D");
PARAM_FLAG("print_accuracy", "Print accuracies on initial and transformed "
    "dataset", "P");
PARAM_FLAG("normalize", "Use a normalized starting point for optimization. It"
    "is useful for when points are far apart, or when SGD is returning NaN.",
    "N");
PARAM_INT_IN("max_iterations", "Maximum number of iterations for "
    "L-BFGS (0 indicates no limit).", "n", 100000);
PARAM_DOUBLE_IN("tolerance", "Maximum tolerance for termination. ", "t", 1e-7);
PARAM_DOUBLE_IN("wTolerance", "Maximum tolerance for termination of binary "
    "search approximation.", "w", 1e-5);
PARAM_DOUBLE_IN("wHigh", "Upper bound for weight. Used for binary search "
    "approximation of weight.", "H", 10.0);
PARAM_DOUBLE_IN("wLow", "Lower bound for weight. Used for binary search "
    "approximation of weight.", "L", 0.0);

using namespace mlpack;
using namespace mlpack::boostmetric;
using namespace mlpack::metric;
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
  neighbor::KNN knn;

  knn.Train(dataset);
  knn.Search(k, neighbors, distances);

  // Keep count.
  size_t count = 0;

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    arma::vec Map;
    Map.zeros(uniqueLabels.n_cols);

    for (size_t j = 0; j < k; j++)
    {
      Map(labels(neighbors(j, i))) +=
          1 / std::pow(distances(j, i) + 1, 2);
    }

    arma::vec index = arma::conv_to<arma::vec>::from(arma::find(Map
        == arma::max(Map)));

    // Increase count if labels match.
    if (index(0) == labels(i))
        count++;
  }

  // return accuracy.
  return ((double) count / dataset.n_cols) * 100;
}

static void mlpackMain()
{
  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");

  RequireParamValue<int>("k", [](int x) { return x > 0; }, true,
      "number of targets must be positive");
  RequireParamValue<int>("max_iterations", [](int x)
      { return x >= 0; }, true,
      "maximum number of iterations must be non-negative");
  RequireParamValue<double>("tolerance",
      [](double x) { return x >= 0.0; }, true,
      "tolerance must be non-negative");
  RequireParamValue<double>("wTolerance",
      [](double x) { return x >= 0.0; }, true,
      "weight tolerance must be non-negative");
  RequireParamValue<double>("wHigh",
      [](double x) { return x >= 0.0; }, true,
      "upper bound for weight must be non-negative");
  RequireParamValue<double>("wLow",
      [](double x) { return x >= 0.0; }, true,
      "lower bound for weight must be non-negative");

  const size_t k = (size_t) CLI::GetParam<int>("k");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const double wTolerance = CLI::GetParam<double>("wTolerance");
  const double wHigh = CLI::GetParam<double>("wHigh");
  const double wLow = CLI::GetParam<double>("wLow");
  const bool normalize = CLI::HasParam("normalize");
  const bool printAccuracy = CLI::HasParam("print_accuracy");
  
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

  // Now create the BoostMetric object and perform the optimization.
  BOOSTMETRIC<> bm(data, labels, k);
  bm.MaxIterations() = maxIterations;
  bm.Tolerance() = tolerance;
  bm.WTolerance() = wTolerance;
  bm.WHigh() = wHigh;
  bm.WLow() = wLow;
  bm.LearnDistance(distance);

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
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = distance;
  if (CLI::HasParam("transformed_data"))
    CLI::GetParam<arma::mat>("transformed_data") = std::move(distance * data);
}
