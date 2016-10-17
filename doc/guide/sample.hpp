/*! @page sample Simple Sample mlpack Programs

@section sampleintro Introduction

On this page, several simple mlpack examples are contained, in increasing order
of complexity.

@section covariance Covariance Computation

A simple program to compute the covariance of a data matrix ("data.csv"),
assuming that the data is already centered, and save it to file.

@code
// Includes all relevant components of mlpack.
#include <mlpack/core.hpp>

// Convenience.
using namespace mlpack;

int main()
{
  // First, load the data.
  arma::mat data;
  // Use data::Load() which transposes the matrix.
  data::Load("data.csv", data, true);

  // Now compute the covariance.  We assume that the data is already centered.
  // Remember, because the matrix is column-major, the covariance operation is
  // transposed.
  arma::mat cov = data * trans(data) / data.n_cols;

  // Save the output.
  data::Save("cov.csv", cov, true);
}
@endcode

@section nn Nearest Neighbor

This simple program uses the mlpack::neighbor::NeighborSearch object to find the
nearest neighbor of each point in a dataset using the L1 metric, and then print
the index of the neighbor and the distance of it to stdout.

@code
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance

int main()
{
  // Load the data from data.csv (hard-coded).  Use CLI for simple command-line
  // parameter handling.
  arma::mat data;
  data::Load("data.csv", data, true);

  // Use templates to specify that we want a NeighborSearch object which uses
  // the Manhattan distance.
  NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(data);

  // Create the object we will store the nearest neighbors in.
  arma::Mat<size_t> neighbors;
  arma::mat distances; // We need to store the distance too.

  // Compute the neighbors.
  nn.Search(1, neighbors, distances);

  // Write each neighbor and distance using Log.
  for (size_t i = 0; i < neighbors.n_elem; ++i)
  {
    Log::Info << "Nearest neighbor of point " << i << " is point "
        << neighbors[i] << " and the distance is " << distances[i] << ".\n";
  }
}
@endcode

@section other Other examples

For more complex examples, it is useful to refer to the main executables:

 - methods/neighbor_search/knn_main.cpp
 - methods/neighbor_search/kfn_main.cpp
 - methods/emst/emst_main.cpp
 - methods/radical/radical_main.cpp
 - methods/nca/nca_main.cpp
 - methods/naive_bayes/nbc_main.cpp
 - methods/pca/pca_main.cpp
 - methods/lars/lars_main.cpp
 - methods/linear_regression/linear_regression_main.cpp
 - methods/gmm/gmm_main.cpp
 - methods/kmeans/kmeans_main.cpp

*/
