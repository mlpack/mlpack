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
PARAM_DOUBLE("step_size", "Step size for stochastic gradient descent.", "s",
    0.01);
PARAM_INT("max_iterations", "Maximum number of iterations for stochastic "
    "gradient descent (0 indicates no limit).", "n", 500000);
PARAM_DOUBLE("tolerance", "Maximum tolerance for termination of stochastic "
    "gradient descent.", "t", 1e-7);
PARAM_FLAG("no_normalization", "Do not normalize distances (this should not be"
    "set if squared distances between points are greater than 700).", "N");

using namespace mlpack;
using namespace mlpack::nca;
using namespace mlpack::metric;
using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
  // Parse command line.
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string outputFile = CLI::GetParam<string>("output_file");

  const double stepSize = CLI::GetParam<double>("step_size");
  const size_t maxIterations = CLI::GetParam<int>("max_iterations");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const bool normalize = !CLI::HasParam("no_normalization");

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

  // Now create the NCA object and run the optimization.
  NCA<LMetric<2> > nca(data, labels.unsafe_col(0), stepSize, maxIterations,
      tolerance, normalize);

  mat distance;
  nca.LearnDistance(distance);

  // Save the output.
  data::Save(CLI::GetParam<string>("output_file").c_str(), distance, true);
}
