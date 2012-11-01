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
    "file (--labels_file).");

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
PARAM_FLAG("normalize", "Normalize data; useful for datasets where points are "
    "far apart, or when SGD is converging to an objective of NaN.", "N");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

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

  // Normalize the data, if necessary.
  if (normalize)
  {
    // Find the minimum and maximum values for each dimension.
    arma::vec range = arma::max(data, 1) - arma::min(data, 1);

    // Now find the maximum range.
    double maxRange = arma::max(range);

    // We can place a (lazy) upper bound on the distance with range^2 * d.
    // Since we want no distance greater than 700 (because std::exp(-750)
    // underflows), we can normalize with (range^2 * d) / 700).
    double normalization = (std::pow(maxRange, 2.0) * data.n_rows) / 700.0;
    data /= normalization; // Element-wise division.

    Log::Info << "Data normalized (normalization constant " << normalization
        << ")." << std::endl;
  }

  // Now create the NCA object and run the optimization.
  NCA<LMetric<2> > nca(data, labels.unsafe_col(0), stepSize, maxIterations,
      tolerance);

  mat distance;
  nca.LearnDistance(distance);

  // Save the output.
  data::Save(CLI::GetParam<string>("output_file").c_str(), distance, true);
}
