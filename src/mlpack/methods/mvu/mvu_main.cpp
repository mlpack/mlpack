/**
 * @file mvu_main.cpp
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
#include <mlpack/core.hpp>
#include "mvu.hpp"

PROGRAM_INFO("Maximum Variance Unfolding (MVU)", "This program implements "
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
using namespace arma;
using namespace std;

int main(int argc, char **argv)
{
  // Read from command line.
  CLI::ParseCommandLine(argc, argv);
  const string inputFile = CLI::GetParam<string>("input_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const int newDim = CLI::GetParam<int>("new_dim");
  const int numNeighbors = CLI::GetParam<int>("num_neighbors");

  if (!CLI::HasParam("output"))
    Log::Warn << "--output_file (-o) is not specified; no results will be "
        << "saved!" << endl;

  RandomSeed(time(NULL));

  // Load input dataset.
  mat data = std::move(CLI::GetParam<arma::mat>("input"));

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
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(output);
}
