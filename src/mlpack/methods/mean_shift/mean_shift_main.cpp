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
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME mean_shift

#include <mlpack/core/util/mlpack_main.hpp>

#include "mean_shift.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Mean Shift Clustering");

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
    "https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf"
    "&doi=1c168275c59ba382588350ee1443537f59978183");
BINDING_SEE_ALSO("mlpack::mean_shift::MeanShift C++ class documentation",
    "@doc/user/methods/mean_shift.md");

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

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  const double radius = params.Get<double>("radius");
  const int maxIterations = params.Get<int>("max_iterations");

  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "maximum iterations must be greater than or equal to 0");

  // Make sure we have an output file if we're not doing the work in-place.
  RequireAtLeastOnePassed(params, { "in_place", "output", "centroid" }, false,
      "no results will be saved");
  ReportIgnoredParam(params, {{ "output", false }}, "labels_only");
  ReportIgnoredParam(params, {{ "in_place", true }}, "output");
  ReportIgnoredParam(params, {{ "in_place", true }}, "labels_only");

  arma::mat dataset = std::move(params.Get<arma::mat>("input"));
  arma::mat centroids;
  arma::Row<size_t> assignments;

  MeanShift<> meanShift(radius, maxIterations);

  timers.Start("clustering");
  Log::Info << "Performing mean shift clustering..." << endl;
  meanShift.Cluster(dataset, assignments, centroids,
      params.Has("force_convergence"));
  timers.Stop("clustering");

  Log::Info << "Found " << centroids.n_cols << " centroids." << endl;
  if (radius <= 0.0)
    Log::Info << "Estimated radius was " << meanShift.Radius() << ".\n";

  if (params.Has("in_place"))
  {
    // Add the column of assignments to the dataset; but we have to convert them
    // to type double first.
    arma::vec converted(assignments.n_elem);
    for (size_t i = 0; i < assignments.n_elem; ++i)
      converted(i) = (double) assignments(i);

    dataset.insert_rows(dataset.n_rows, trans(converted));

    // Save the dataset.
    params.MakeInPlaceCopy("output", "input");
    params.Get<arma::mat>("output") = std::move(dataset);
  }
  else if (params.Has("output"))
  {
    if (!params.Has("labels_only"))
    {
      // Convert the assignments to doubles.
      arma::vec converted(assignments.n_elem);
      for (size_t i = 0; i < assignments.n_elem; ++i)
        converted(i) = (double) assignments(i);

      dataset.insert_rows(dataset.n_rows, trans(converted));

      // Now save, in the different file.
      params.Get<arma::mat>("output") = std::move(dataset);
    }
    else
    {
      // TODO: figure out how to output as an arma::Mat<size_t> so that files
      // aren't way larger than needed.
      params.Get<arma::mat>("output") =
          ConvTo<arma::mat>::From(assignments);
    }
  }

  // Should we write the centroids to a file?
  if (params.Has("centroid"))
    params.Get<arma::mat>("centroid") = std::move(centroids);
}
