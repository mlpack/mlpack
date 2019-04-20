/**
 * @file isomap_main.cpp
 * @author Rishabh Ranjan
 * 
 * Main executable to run Isomap.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "isomap.hpp"

using namespace mlpack;
using namespace mlpack::isomap;
using namespace mlpack::util;
using namespace std;

//Document program
PROGRAM_INFO("Isomap",
    // Short Description
    "An implementation of Isomap algorithm for dimensionality reduction. "
    "Given an input dataset the algorithm fits the dataset into a lower "
    "dimensional space using the k-nearest-neighbor graph and "
    "multidimensional scaling.",
    // Long Description
    "An implementation of Isomap algorithm which is part of Manifold Learning "
    "algorithms, which are used for non-linear dimensionality reduction. "
    "Given a dataset, which is assumed to have data points embedded in a"
    "higher dimesional space, it gives as output the dataset in a lower "
    "dimensional space, thus reducing dimensions of the data."
    "\n"
    "The algorithm first forms a distance matrix from input dataset, finds "
    "nearest neighbors, finds shortest distance for all pairs of points in "
    "dataset and then using this distance matrix finds lower dimesional "
    "points whose interpoint distances correspond to this distance matrix, "
    "using classical Multidimensional Scaling."
    "\n\n"
    "Use the " + PRINT_PARAM_STRING("input") + " parameter to specify the "
    "dataset to perform Isomap on."
    "\n"
    "Use the " + PRINT_PARAM_STRING("neighbors") + " parameter to specify "
    "the number of nearest neighbors to consider for constructing the "
    "neighbourhood graph (make sure it is enough so that a connected "
    "graph is formed)."
    "\n"
    "Use the "+ PRINT_PARAM_STRING("new_dimesionality") + " parameter to "
    "specify the number of dimensions to be kept in the output."
    "\n\n"
    "For example, command to reduce the dimensions of dataset " +
    PRINT_DATASET("data") + " to 5 dimensions, using 4 neighbors to "
    "construct the k-nearest-neighbor graph, and store it in the matrix "
    + PRINT_DATASET("output") + "would be:"
    "\n\n" +
    PRINT_CALL("isomap", "input", "data", "new_dimensionality", 5,
        "neighbors", 4, "output", "output"),
    SEE_ALSO("Isomap on Wikipedia","https://en.wikipedia.org/wiki/Isomap"));

// Parameters for the program
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform Isomap on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_INT_IN("neighbors", "The number of neighbors to consider for "
    "k-nearest.", "k", 3);
PARAM_INT_IN("new_dimensionality", "The number of dimensions in the output",
    "d", 0);

static void mlpackMain()
{
  // Load input dataset.
  arma::mat dataset = std::move(CLI::GetParam<arma::mat>("input"));

  // Issue a warning if the user did not specify an output file.
  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");

  // The number of neighbors to consider for k-nearest
  RequireParamValue<int>("neighbors", [](int x) { return x > 0; },
      true, "number of neighbors must be greater than zero");
  std::ostringstream error;
  error << "cannot be greater than or equal to number of data points ("
      << dataset.n_cols << ")";
  RequireParamValue<int>("neighbors",
      [dataset](int x) { return x < (int) dataset.n_cols; }, true,
      error.str());

  size_t k = CLI::GetParam<int>("neighbors");

  // The number of dimensions to include in output.
  RequireParamValue<int>("new_dimensionality",
      [](int x) { return x >= 0; }, true,
      "number of output dimensions must be non-negetive");

  int d = CLI::GetParam<int>("new_dimensionality");

  // Running isomap with the specified parameters (this call should be
  // changed while including further variations)
  Isomap<KNearest, Dijkstra> iso(k, d);
  iso.Apply(dataset);

  // Save output if output parameter is specified
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(dataset);
}
