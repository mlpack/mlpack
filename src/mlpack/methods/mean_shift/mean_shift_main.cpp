/**
 * @file methods/mean_shift/mean_shift_main.cpp
 * @author Shangtong Zhang
 *
 * Executable for running Mean Shift.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include "mean_shift.hpp"

using namespace mlpack;
using namespace mlpack::meanshift;
using namespace mlpack::kernel;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_NAME("Mean Shift Clustering");

// Short description.
BINDING_SHORT_DESC(
    "A fast implementation of mean-shift clustering using dual-tree range "
    "search.  Given a dataset, this uses the mean shift algorithm to produce "
    "and return a clustering of the data.");

// Long description.
BINDING_LONG_DESC(
    "This program performs mean shift clustering on the given dataset, storing "
    "the learned cluster assignments either as a column of labels in the input "
    "dataset or separately."
    "\n\n"
    "The input dataset should be specified with the " +
    PRINT_PARAM_STRING("input") + " parameter, and the radius used for search"
    " can be specified with the " + PRINT_PARAM_STRING("radius") + " "
    "parameter.  The maximum number of iterations before algorithm termination "
    "is controlled with the " + PRINT_PARAM_STRING("max_iterations") + " "
    "parameter."
    "\n\n"
    "The output labels may be saved with the " + PRINT_PARAM_STRING("output") +
    " output parameter and the centroids of each cluster may be saved with the"
    " " + PRINT_PARAM_STRING("centroid") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to run mean shift clustering on the dataset " +
    PRINT_DATASET("data") + " and store the centroids to " +
    PRINT_DATASET("centroids") + ", the following command may be used: "
    "\n\n" +
    PRINT_CALL("mean_shift", "input", "data", "centroid", "centroids"));

// See also...
BINDING_SEE_ALSO("@kmeans", "#kmeans");
BINDING_SEE_ALSO("@dbscan", "#dbscan");
BINDING_SEE_ALSO("Mean shift on Wikipedia",
        "https://en.wikipedia.org/wiki/Mean_shift");
BINDING_SEE_ALSO("Mean Shift, Mode Seeking, and Clustering (pdf)",
        "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.510.1222"
        "&rep=rep1&type=pdf");
BINDING_SEE_ALSO("mlpack::mean_shift::MeanShift C++ class documentation",
        "@doxygen/classmlpack_1_1meanshift_1_1MeanShift.html");

// Required options.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform clustering on.", "i");

// Output options.
PARAM_FLAG("in_place", "If specified, a column containing the learned cluster "
    "assignments will be added to the input dataset file.  In this case, "
    "--output_file is overridden.  (Do not use with Python.)", "P");
PARAM_FLAG("labels_only", "If specified, only the output labels will be "
    "written to the file specified by --output_file.", "l");
PARAM_FLAG("force_convergence", "If specified, the mean shift algorithm will "
  "continue running regardless of max_iterations until the clusters converge."
  , "f");
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

static void mlpackMain()
{
  const double radius = IO::GetParam<double>("radius");
  const int maxIterations = IO::GetParam<int>("max_iterations");

  RequireParamValue<int>("max_iterations", [](int x) { return x >= 0; }, true,
      "maximum iterations must be greater than or equal to 0");

  // Make sure we have an output file if we're not doing the work in-place.
  RequireAtLeastOnePassed({ "in_place", "output", "centroid" }, false,
      "no results will be saved");
  ReportIgnoredParam({{ "output", false }}, "labels_only");
  ReportIgnoredParam({{ "in_place", true }}, "output");
  ReportIgnoredParam({{ "in_place", true }}, "labels_only");

  arma::mat dataset = std::move(IO::GetParam<arma::mat>("input"));
  arma::mat centroids;
  arma::Row<size_t> assignments;

  MeanShift<> meanShift(radius, maxIterations);

  Timer::Start("clustering");
  Log::Info << "Performing mean shift clustering..." << endl;
  meanShift.Cluster(dataset, assignments, centroids,
    IO::HasParam("force_convergence"));
  Timer::Stop("clustering");

  Log::Info << "Found " << centroids.n_cols << " centroids." << endl;
  if (radius <= 0.0)
    Log::Info << "Estimated radius was " << meanShift.Radius() << ".\n";

  if (IO::HasParam("in_place"))
  {
    // Add the column of assignments to the dataset; but we have to convert them
    // to type double first.
    arma::vec converted(assignments.n_elem);
    for (size_t i = 0; i < assignments.n_elem; ++i)
      converted(i) = (double) assignments(i);

    dataset.insert_rows(dataset.n_rows, trans(converted));

    // Save the dataset.
    IO::MakeInPlaceCopy("output", "input");
    IO::GetParam<arma::mat>("output") = std::move(dataset);
  }
  else if (IO::HasParam("output"))
  {
    if (!IO::HasParam("labels_only"))
    {
      // Convert the assignments to doubles.
      arma::vec converted(assignments.n_elem);
      for (size_t i = 0; i < assignments.n_elem; ++i)
        converted(i) = (double) assignments(i);

      dataset.insert_rows(dataset.n_rows, trans(converted));

      // Now save, in the different file.
      IO::GetParam<arma::mat>("output") = std::move(dataset);
    }
    else
    {
      // TODO: figure out how to output as an arma::Mat<size_t> so that files
      // aren't way larger than needed.
      IO::GetParam<arma::mat>("output") =
          arma::conv_to<arma::mat>::from(assignments);
    }
  }

  // Should we write the centroids to a file?
  if (IO::HasParam("centroid"))
    IO::GetParam<arma::mat>("centroid") = std::move(centroids);
}
