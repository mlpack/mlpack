/**
 * @file knn_main.cpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * A lite Implementation of the kNN executable.  Allows some number of standard
 * options.
 * This implementation is intended to be used on low resource devices
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


/*  CLI11 library headers */
#include <CLI/CLI.hpp>

#include "neighbor_search.hpp"
#include "ns_model.hpp"
#include "unmap.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

int main(int argc, char** argv)
{
   CLI::App app{
  "k-Nearest-Neighbors Search",
  // Short description.
  "A lite implementation of k-nearest-neighbor search using "
  "dual-tree algorithms.  Given a set of reference points and query points, "
  "this can find the k nearest neighbors in the reference set of each query "
  "point using trees; trees that are built can be saved for future use."
  }; 
  arma::mat referenceSet;
  arma::mat queryData;
  size_t k = 0;

  app.add_option("-k, --K-Nearest-Neighbor",
                  k,
                  "Number of nearest neighbors to find."); 
  app.add_option("-r, --referenceSet",
                 referenceSet,
                 " Reference set to be used.");
  app.add_option("-r, --queryData",
                 queryData,
                 " Matrix containing query points");
  
  CLI11_PARSE(app, argc, argv);  

  NeighborSearch<NearestNeighborSort> knn(std::move(referenceSet), DUAL_TREE_MODE);


  // Log::Info << "Loaded kNN model from '"
  //           << CMD::GetPrintableParam<KNNModel*>("input_model")
  //           << "' (trained on " << knn->Dataset().n_rows << "x"
  //           << knn->Dataset().n_cols << " dataset)." << endl;

      if (!queryData.empty()) {
        if (queryData.n_rows != referenceSet.n_rows) {
         const size_t dimensions = referenceSet.n_rows;
         if (!referenceSet.empty())
         std::cerr << "Query has invalid dimensions(" << queryData.n_rows
                   << "); should be " << dimensions << "!" << endl;
      }
    }

    // Sanity check on k value: must be greater than 0, must be less than or
    // equal to the number of reference points.  Since it is unsigned,
    // we only test the upper bound.
    if (k > referenceSet.n_cols) {
      // Clean memory if needed before crashing.
      const size_t referencePoints = referenceSet.n_cols;

      // Sanity check on k value: must not be equal to the number of reference
      // points when query data has not been provided.
      if (queryData.empty() && k == referenceSet.n_cols) {
        const size_t referencePoints = referenceSet.n_cols;
        if (!referenceSet.empty())
           std::cerr << "Invalid k: " << k << "; must be less than the number of "
                     << "reference points (" << referencePoints
                     << ") if query data has "
                     << "not been provided." << endl;
      }
    }

      // Now run the search.
      arma::Mat<size_t> neighbors;
      arma::mat distances;

      if (CMD::HasParam("query"))
        knn.Search(std::move(queryData), k, neighbors, distances);
      else
        knn.Search(k, neighbors, distances);
      std::cout << "Search complete." << endl;

  //     // Save output.
  //     CMD::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
  //     CMD::GetParam<arma::mat>("distances") = std::move(distances);
  //   }
  // }
}
