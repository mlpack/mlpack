/**
 * @file isomap_main.cpp
 * @author Rishabh Ranjan
 * 
 * Main executable to run Isomap
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

PROGRAM_INFO("Isomap",
    // Short Description
    "An implementation of Isomap algorithm for dimensionality reduction. "
    "Given an input dataset the algorithm fits the dataset into a lower "
    "dimensional space, assuming the data is embedded into a "
    "higher dimensional space but belongs to a lower dimensional space.",
    // Long Description
    "An implementation of Isomap algorithm which is part of Manifold Learning "
    "algorithms, which are used for dimensionality reduction. "
    "Given a dataset, which is assumed to have data points embedded in a"
    "higher dimesional space, it gives as output the dataset in a lower "
    "dimensional space, thus reducing dimensions of the data.\n\n"
    "Use the " + PRINT_PARAM_STRING("input") + " parameter to specify the "
    "dataset to perform Isomap on.\n Use the " +
    PRINT_PARAM_STRING("neighbours") + " parameter to specify the number "
    "of nearest neighbours to consider for constructing the neighbourhood "
    "graph (make sure it is enough so that a connected graph is formed).\n "
);

// Parameters for the program
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform Isomap on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_INT_IN("neighbours", "The number of neighbours to consider for "
    "k-nearest.", "n", 3);

static void mlpackMain()
{
  // Load input dataset.
  arma::mat& input = CLI::GetParam<arma::mat>("input");

  // Issue a warning if the user did not specify an output file.
  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");

  // The number of neighbours to consider for k-nearest
  RequireParamValue<int>("neighbours", [](int x) { return x > 0; },
      true, "number of neighbours must be greater than zero");
  std::ostringstream error;
  error << "cannot be greater than or equal to number of data points ("
      << input.n_cols << ")";
  RequireParamValue<int>("neighbours",
      [input](int x) { return x < (int) input.n_cols; }, true,
      error.str());

  size_t n_neighbours = CLI::GetParam<int>("neighbours");

  // Running isomap with the specified parameters (this call should be
  // changed while including further variation)
  Isomap<K_Nearest, Dijkstra> iso(n_neighbours);
  iso.Apply(input);

  // Save output if output parameter is specified
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(input);
}
