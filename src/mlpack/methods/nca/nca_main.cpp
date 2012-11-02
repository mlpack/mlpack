/**
 * @file nca_main.cpp
 * @author Ryan Curtin
 *
 * Executable for Neighborhood Components Analysis.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include "nca.hpp"

// Define parameters.
PROGRAM_INFO("Neighborhood Components Analysis (NCA)",
    "This program implements Neighborhood Components Analysis, both a linear "
    "dimensionality reduction technique and a distance learning technique.  The"
    " method seeks to improve k-nearest-neighbor classification on a dataset "
    "by scaling the dimensions.  The method is nonparametric, and does not "
    "require a value of k.  It works by using stochastic (\"soft\") neighbor "
    "assignments and using optimization techniques over the gradient of the "
    "accuracy of the neighbor assignments.\n"
    "\n"
    "To work, this algorithm needs labeled data.  It can be given as the last "
    "row of the input dataset (--input_file), or alternatively in a separate "
    "file (--labels_file).\n"
    "\n"
    "This implementation of NCA uses stochastic gradient descent, which depends"
    " primarily on two parameters: the step size (--step_size) and the maximum "
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
    "between objectives for SGD to terminate.  Be careful -- setting the "
    "tolerance instead of the maximum iterations can take a very long time and "
    "may actually never converge due to the properties of the SGD optimizer.");

PARAM_STRING_REQ("input_file", "Input dataset to run NCA on.", "i");
PARAM_STRING_REQ("output_file", "Output file for learned distance matrix.",
    "o");
PARAM_STRING("labels_file", "File of labels for input dataset.", "l", "");
PARAM_DOUBLE("step_size", "Step size for stochastic gradient descent (alpha).",
    "a", 0.01);
PARAM_INT("max_iterations", "Maximum number of iterations for stochastic "
    "gradient descent (0 indicates no limit).", "n", 500000);
PARAM_DOUBLE("tolerance", "Maximum tolerance for termination of stochastic "
    "gradient descent.", "t", 1e-7);
PARAM_FLAG("normalize", "Use a normalized starting point for optimization. This"
    " is useful for when points are far apart, or when SGD is returning NaN.",
    "N");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_FLAG("linear_scan", "Don't shuffle the order in which data points are "
    "visited for SGD.", "L");

using namespace mlpack;
using namespace mlpack::nca;
using namespace mlpack::metric;
using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
  // Parse command line.
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  const string inputFile = CLI::GetParam<string>("input_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string outputFile = CLI::GetParam<string>("output_file");

  const double stepSize = CLI::GetParam<double>("step_size");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const bool normalize = CLI::HasParam("normalize");
  const bool shuffle = !CLI::HasParam("linear_scan");

  // Load data.
  mat data;
  data::Load(inputFile.c_str(), data, true);

  // Do we want to load labels separately?
  umat labels(data.n_cols, 1);
  if (labelsFile != "")
  {
    data::Load(labelsFile.c_str(), labels, true);

    if (labels.n_rows == 1)
      labels = trans(labels);

    if (labels.n_cols > 1)
      Log::Fatal << "Labels must have only one column or row!" << endl;
  }
  else
  {
    for (size_t i = 0; i < data.n_cols; i++)
      labels[i] = (int) data(data.n_rows - 1, i);

    data.shed_row(data.n_rows - 1);
  }

  mat distance;

  // Normalize the data, if necessary.
  if (normalize)
  {
    // Find the minimum and maximum values for each dimension.
    arma::vec ranges = arma::max(data, 1) - arma::min(data, 1);
    for (size_t d = 0; d < ranges.n_elem; ++d)
      if (ranges[d] == 0.0)
        ranges[d] = 1; // A range of 0 produces NaN later on.

    distance = diagmat(1.0 / ranges);
    Log::Info << "Using normalized starting point for SGD." << std::endl;
  }
  else
  {
    distance.eye();
  }

  // Now create the NCA object and run the optimization.
  NCA<LMetric<2> > nca(data, labels.unsafe_col(0), stepSize, maxIterations,
      tolerance, shuffle);

  nca.LearnDistance(distance);

  // Save the output.
  data::Save(CLI::GetParam<string>("output_file").c_str(), distance, true);
}
