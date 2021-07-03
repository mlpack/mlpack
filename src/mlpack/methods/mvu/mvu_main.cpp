/**
 * @file methods/mvu/mvu_main.cpp
 * @author Ryan Curtin
 *
 * Executable for MVU.
 *
 * Note: this implementation of MVU does not work.  See #189.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME mvu

#include <mlpack/methods/core/util/mlpack_main.hpp>
#include "mvu.hpp"

// Program Name.
BINDING_USER_NAME("Maximum Variance Unfolding (MVU)");

// Long description.
BINDING_LONG_DESC("This program implements "
    "Maximum Variance Unfolding, a nonlinear dimensionality reduction "
    "technique.  The method minimizes dimensionality by unfolding a manifold "
    "such that the distances to the nearest neighbors of each point are held "
    "constant.");

PARAM_MATRIX_IN_REQ("input", "Input dataset.", "i");
PARAM_INT_IN_REQ("new_dim", "New dimensionality of dataset.", "d");

PARAM_MATRIX_OUT("output", "Matrix to save unfolded dataset to.", "o");
PARAM_INT_IN("num_neighbors", "Number of nearest neighbors to consider while "
    "unfolding.", "k", 5);

using namespace mlpack;
using namespace mlpack::mvu;
using namespace mlpack::math;
using namespace mlpack::util;
using namespace arma;
using namespace std;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers);
{
  const string inputFile = params.Get<string>("input_file");
  const string outputFile = params.Get<string>("output_file");
  const int newDim = params.Get<int>("new_dim");
  const int numNeighbors = params.Get<int>("num_neighbors");

  if (!params.Has("output"))
  {
    Log::Warn << "--output_file (-o) is not specified; no results will be "
        << "saved!" << endl;
  }

  RandomSeed(time(NULL));

  // Load input dataset.
  mat data = std::move(params.Get<arma::mat>("input"));

  // Verify that the requested dimensionality is valid.
  if (newDim <= 0 || newDim > (int) data.n_rows)
  {
    Log::Fatal << "Invalid new dimensionality (" << newDim << ").  Must be "
      << "between 1 and the input dataset dimensionality (" << data.n_rows
      << ")." << std::endl;
  }

  // Verify that the number of neighbors is valid.
  if (numNeighbors <= 0 || numNeighbors > (int) data.n_cols)
  {
    Log::Fatal << "Invalid number of neighbors (" << numNeighbors << ").  Must "
        << "be between 1 and the number of points in the input dataset ("
        << data.n_cols << ")." << std::endl;
  }

  // Now run MVU.
  MVU mvu(data);

  mat output;
  mvu.Unfold(newDim, numNeighbors, output);

  // Save results to file.
  if (params.Has("output"))
    params.Get<arma::mat>("output") = std::move(output);
}
