/**
 * @file mean_shift_main.cpp
 * @author Shangtong Zhang
 *
 * Executable for running Mean Shift.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include "mean_shift.hpp"

using namespace mlpack;
using namespace mlpack::meanshift;
using namespace mlpack::kernel;
using namespace std;

// Define parameters for the executable.
PROGRAM_INFO("Mean Shift Clustering", "This program performs mean shift "
    "clustering on the given dataset, storing the learned cluster assignments "
    "either as a column of labels in the file containing the input dataset or "
    "in a separate file.");

// Required options.
PARAM_MATRIX_IN("input", "Input dataset to perform clustering on.", "i");
// This is kept for reverse compatibility and may be removed in mlpack 3.0.0.
// At that time, --input_file should be made a required parameter.
PARAM_STRING_IN("inputFile", "Input dataset to perform clustering on.", "", "");

// Output options.
PARAM_FLAG("in_place", "If specified, a column containing the learned cluster "
    "assignments will be added to the input dataset file.  In this case, "
    "--output_file is overridden.", "P");
PARAM_FLAG("labels_only", "If specified, only the output labels will be "
    "written to the file specified by --output_file.", "l");
PARAM_MATRIX_OUT("output", "Matrix to write output labels or labeled data to.",
    "o");
PARAM_MATRIX_OUT("centroid", "If specified, the centroids of each cluster will "
    "be written to the given matrix.", "C");

// Mean shift configuration options.
PARAM_INT_IN("max_iterations", "Maximum number of iterations before mean shift "
    "terminates.", "m", 1000);

PARAM_DOUBLE_IN("radius", "If the distance between two centroids is less than "
    "the given radius, one will be removed.  A radius of 0 or less means an "
    "estimate will be calculated and used for the radius.", "r", 0);

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // This is for reverse compatibility and may be removed in mlpack 3.0.0.
  if (CLI::HasParam("inputFile") && CLI::HasParam("input"))
    Log::Fatal << "Cannot specify both --input_file and --inputFile!" << endl;

  if (CLI::HasParam("inputFile"))
  {
    Log::Warn << "--inputFile is deprecated and will be removed in mlpack "
        << "3.0.0; use --input_file instead." << endl;
    CLI::GetUnmappedParam<string>("input_file") =
        CLI::GetParam<string>("inputFile");
  }

  if (!CLI::HasParam("input"))
    Log::Fatal << "--input_file must be specified!" << endl;

  const double radius = CLI::GetParam<double>("radius");
  const int maxIterations = CLI::GetParam<int>("max_iterations");

  if (maxIterations < 0)
  {
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations <<
        ")! Must be greater than or equal to 0." << endl;
  }

  // Make sure we have an output file if we're not doing the work in-place.
  if (!CLI::HasParam("in_place") && !CLI::HasParam("output") &&
      !CLI::HasParam("centroid"))
  {
    Log::Warn << "--output_file, --in_place, and --centroid_file are not set; "
        << "no results will be saved." << endl;
  }

  if (CLI::HasParam("labels_only") && !CLI::HasParam("output"))
    Log::Warn << "--labels_only ignored because --output_file is not specified."
        << endl;

  if (CLI::HasParam("in_place") && CLI::HasParam("output"))
    Log::Warn << "--output_file ignored because --in_place is specified."
        << endl;

  if (CLI::HasParam("in_place") && CLI::HasParam("labels_only"))
    Log::Warn << "--labels_only ignored because --in_place is specified."
        << endl;

  arma::mat dataset = std::move(CLI::GetParam<arma::mat>("input"));
  arma::mat centroids;
  arma::Col<size_t> assignments;

  MeanShift<> meanShift(radius, maxIterations);

  Timer::Start("clustering");
  Log::Info << "Performing mean shift clustering..." << endl;
  meanShift.Cluster(dataset, assignments, centroids);
  Timer::Stop("clustering");

  Log::Info << "Found " << centroids.n_cols << " centroids." << endl;
  if (radius <= 0.0)
    Log::Info << "Estimated radius was " << meanShift.Radius() << ".\n";

  if (CLI::HasParam("in_place"))
  {
    // Add the column of assignments to the dataset; but we have to convert them
    // to type double first.
    arma::vec converted(assignments.n_elem);
    for (size_t i = 0; i < assignments.n_elem; i++)
      converted(i) = (double) assignments(i);

    dataset.insert_rows(dataset.n_rows, trans(converted));

    // Save the dataset.  This takes a little trickery, because we have to set
    // the output matrix parameter to have the same filename associated with it
    // as the input.
    CLI::GetUnmappedParam<arma::mat>("output") =
        CLI::GetUnmappedParam<arma::mat>("input");
    CLI::GetParam<arma::mat>("output") = std::move(dataset);
  }
  else if (CLI::HasParam("output"))
  {
    if (!CLI::HasParam("labels_only"))
    {
      // Convert the assignments to doubles.
      arma::vec converted(assignments.n_elem);
      for (size_t i = 0; i < assignments.n_elem; i++)
        converted(i) = (double) assignments(i);

      dataset.insert_rows(dataset.n_rows, trans(converted));

      // Now save, in the different file.
      CLI::GetParam<arma::mat>("output") = std::move(dataset);
    }
    else
    {
      // We have to add an unsigned matrix output parameter so we can save the
      // labels as the right type.
      CLI::Add<arma::Mat<size_t>>(arma::Mat<size_t>(), "output_labels",
          "Labels for input dataset.", '\0', false, false, true);
      CLI::GetUnmappedParam<arma::Mat<size_t>>("output_labels") =
          CLI::GetUnmappedParam<arma::mat>("output");
      CLI::GetParam<arma::Mat<size_t>>("output_labels") =
          std::move(assignments);
    }
  }

  // Should we write the centroids to a file?
  if (CLI::HasParam("centroid"))
    CLI::GetParam<arma::mat>("centroid") = std::move(centroids);
}
