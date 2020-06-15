/**
 * @file knn_main.cpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Implementation of the kNN executable.  Allows some number of standard
 * options.
 * This implementation is intended to be used on low resource devices
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/prereqs.hpp>

// #include <fstream>
// #include <iostream>
// #include <string>

#include "neighbor_search.hpp"
#include "ns_model.hpp"
#include "unmap.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

// Convenience typedef.
typedef NSModel<NearestNeighborSort> KNNModel;

// Information about the program itself.
PROGRAM_INFO(
  "k-Nearest-Neighbors Search",
  // Short description.
  "An implementation of k-nearest-neighbor search using "
  "dual-tree algorithms.  Given a set of reference points and query points, "
  "this can find the k nearest neighbors in the reference set of each query "
  "point using trees; trees that are built can be saved for future use.",
  // Long description.
  "This program will calculate the k-nearest-neighbors of a set of "
  "points using kd-trees or cover trees (cover tree support is experimental "
  "and may be slow). You may specify a separate set of "
  "reference points and query points, or just a reference set which will be "
  "used as both the reference and query set."
  "\n\n"
  "For example, the following command will calculate the 5 nearest neighbors "
  "of each point in " +
    PRINT_DATASET("input") +
    " and store the distances "
    "in " +
    PRINT_DATASET("distances") + " and the neighbors in " +
    PRINT_DATASET("neighbors") +
    ": "
    "\n\n" +
    PRINT_CALL("knn",
               "k",
               5,
               "reference",
               "input",
               "neighbors",
               "neighbors",
               "distances",
               "distances") +
    "\n\n"
    "The output is organized such that row i and column j in the neighbors "
    "output matrix corresponds to the index of the point in the reference set "
    "which is the j'th nearest neighbor from the point in the query set with "
    "index i.  Row j and column i in the distances output matrix corresponds to"
    " the distance between those two points.",
  SEE_ALSO("NeighborSearch tutorial (k-nearest-neighbors)",
           "@doxygen/nstutorial.html"),
  SEE_ALSO("Tree-independent dual-tree algorithms (pdf)",
           "http://proceedings.mlr.press/v28/curtin13.pdf"),
  SEE_ALSO("mlpack::neighbor::NeighborSearch C++ class documentation",
           "@doxygen/classmlpack_1_1neighbor_1_1NeighborSearch.html"));

 // Define our input parameters that this program will take.
 PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
 PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");

 // // The option exists to load or save models.
//PARAM_MODEL_IN(KNNModel, "input_model", "Pre-trained kNN model.", "m");

 // The user may specify a query file of query points and a number of nearest
 // neighbors to search for.
 PARAM_MATRIX_IN("query", "Matrix containing query points (optional).", "q");
 PARAM_INT_IN("k", "Number of nearest neighbors to find.", "k", 0);

static void
mlpackMain()
{
  ReportIgnoredParam({ { "input_model", true } }, "tree_type");
  ReportIgnoredParam({ { "input_model", true } }, "random_basis");

  // The user should give something to do...
  RequireAtLeastOnePassed(
    { "k", "output_model" }, false, "no results will be saved");

  // If the user specifies k but no output files, they should be warned.
  if (CMD::HasParam("k")) {
    RequireAtLeastOnePassed(
      { "neighbors", "distances" },
      false,
      "nearest neighbor search results will not be saved");
  }

  // If the user specifies output files but no k, they should be warned.
  ReportIgnoredParam({ { "k", false } }, "neighbors");
  ReportIgnoredParam({ { "k", false } }, "distances");
  ReportIgnoredParam({ { "k", false } }, "query");
  KNNModel* knn;

  // Load the model from file.
 knn = CMD::GetParam<KNNModel*>("input_model");

  // Adjust search mode.
  // Use the dual tree mode as a default mode  
  knn->SearchMode() = DUAL_TREE_MODE;

  Log::Info << "Loaded kNN model from '"
            << CMD::GetPrintableParam<KNNModel*>("input_model")
            << "' (trained on " << knn->Dataset().n_rows << "x"
            << knn->Dataset().n_cols << " dataset)." << endl;

  // Perform search, if desired.
  if (CMD::HasParam("k")) {
    const size_t k = (size_t)CMD::GetParam<int>("k");

    arma::mat queryData;
    if (CMD::HasParam("query")) {
      Log::Info << "Using query data from "
                << CMD::GetPrintableParam<arma::mat>("query") << "." << endl;
      queryData = std::move(CMD::GetParam<arma::mat>("query"));
      if (queryData.n_rows != knn->Dataset().n_rows) {
        // Clean memory if needed before crashing.
        const size_t dimensions = knn->Dataset().n_rows;
        if (CMD::HasParam("reference"))
          delete knn;
        Log::Fatal << "Query has invalid dimensions(" << queryData.n_rows
                   << "); should be " << dimensions << "!" << endl;
      }
    }

    // Sanity check on k value: must be greater than 0, must be less than or
    // equal to the number of reference points.  Since it is unsigned,
    // we only test the upper bound.
    if (k > knn->Dataset().n_cols) {
      // Clean memory if needed before crashing.
      const size_t referencePoints = knn->Dataset().n_cols;

      // Sanity check on k value: must not be equal to the number of reference
      // points when query data has not been provided.
      if (!CMD::HasParam("query") && k == knn->Dataset().n_cols) {
        // Clean memory if needed before crashing.
        const size_t referencePoints = knn->Dataset().n_cols;
        if (CMD::HasParam("reference"))
          delete knn;
        Log::Fatal << "Invalid k: " << k << "; must be less than the number of "
                   << "reference points (" << referencePoints
                   << ") if query data has "
                   << "not been provided." << endl;
      }

      // Now run the search.
      arma::Mat<size_t> neighbors;
      arma::mat distances;

      if (CMD::HasParam("query"))
        knn->Search(std::move(queryData), k, neighbors, distances);
      else
        knn->Search(k, neighbors, distances);
      Log::Info << "Search complete." << endl;

      // Save output.
      CMD::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
      CMD::GetParam<arma::mat>("distances") = std::move(distances);
    }
  }
}
