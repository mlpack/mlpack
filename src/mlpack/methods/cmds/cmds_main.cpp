/**
 * @file cmds_main.cpp
 * @author Rishabh Ranjan
 * 
 * Main executable to run cMDS (classical Multidimensional Scaling).
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "cmds.hpp"

using namespace mlpack;
using namespace mlpack::cmds;
using namespace mlpack::util;
using namespace std;

// Document program
PROGRAM_INFO("Classical Multidimensional Scaling",
    // Short Description
    "An implementation of classical Multidimensional Scaling, which "
    "performs non-linear dimensionality reduction on a given dataset or "
    "finds d-dimensional points to represent a given dissimilarity matrix.",
    // Long Description
    "Classical Multidimensional Scaling, also knows as Principal Coordinates "
    "Analysis (PCoA), is an algorithm to represent a dissimilarity matrix "
    "d-dimensional points whose interpoint distances correspond to the "
    "dissimilarity matrix. It is also used for non-linear dimensionality "
    "reduction by taking the input dataset, calculating their interpoint "
    "distances and using this dissimilarity matrix to find lower dimensional "
    "points to represent the dataset."
    "\n\n"
    "Use the " + PRINT_PARAM_STRING("input") + " parameter to specify the "
    "dissimilarity matrix or dataset to perform cMDS on. While supplying "
    "dissimilarity matrix make sure it is symmetric."
    "\n"
    "Use the " + PRINT_PARAM_STRING("dimesions") + " parameter to specify "
    "the number of dimensions in output points."
    "\n"
    "Use the " + PRINT_PARAM_STRING("calc_dissimilarity") + " parameter to "
    "specify if the input is already a dissimilarity matrix or it needs to "
    "be calculated (in case the input is a dataset)."
    "\n\n"
    "For example, to reduce the dimensionality of dataset " +
    PRINT_DATASET("data") + "to 5 dimensions and storing the output in " +
    PRINT_DATASET("output") + ", use the following command:" +
    PRINT_CALL("cmds", "input", "data", "dimensions", 5,
        "calc_dimensionality", "true", "output", "output"),
    SEE_ALSO("Multidimensional Scaling on Wikipedia",
        "https://en.wikipedia.org/wiki/Multidimensional_scaling"));

// Parameters for the program
PARAM_MATRIX_IN_REQ("input", "Input to perform cMDS on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_INT_IN("dimensions", "The number of dimensions to restrict "
    "output to. If 0, the output would contain maximum number of "
    "dimensions the points can have in Euclidean space.", "d", 0);
PARAM_FLAG("calc_dissimilarity", "Whether to calculate dissimilarity "
    "matrix(true) or is it precomputed(false).", "c");

static void mlpackMain()
{
  // Load input dataset.
  arma::mat data = std::move(CLI::GetParam<arma::mat>("input"));

  // Issue a warning if the user did not specify an output file.
  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");

  // The number of dimensions to include in output.
  RequireParamValue<int>("dimensions",
      [](int x) { return x >= 0; }, true,
      "number of output dimensions must be non-negetive");

  bool cd = CLI::GetParam<bool>("calc_dissimilarity");
  int d = CLI::GetParam<int>("dimensions");

  // Running cMDS with the specified parameters (this call should be
  // changed while including further variations)

  Cmds c;
  c.Apply(cd, d, data);

  // Save output if output parameter is specified
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(data);
}
