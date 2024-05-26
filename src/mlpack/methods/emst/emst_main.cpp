/**
 * @file methods/emst/emst_main.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Calls the DualTreeBoruvka algorithm from dtb.hpp.
 * Can optionally call naive Boruvka's method.
 *
 * For algorithm details, see:
 *
 * @code
 * @inproceedings{
 *   author = {March, W.B., Ram, P., and Gray, A.G.},
 *   title = {{Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,
 *      Applications.}},
 *   booktitle = {Proceedings of the 16th ACM SIGKDD International Conference
 *      on Knowledge Discovery and Data Mining}
 *   series = {KDD 2010},
 *   year = {2010}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME emst

#include <mlpack/core/util/mlpack_main.hpp>

#include "dtb.hpp"

// Program Name.
BINDING_USER_NAME("Fast Euclidean Minimum Spanning Tree");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the Dual-Tree Boruvka algorithm for computing the "
    "Euclidean minimum spanning tree of a set of input points.");

// Long description.
BINDING_LONG_DESC(
    "This program can compute the Euclidean minimum spanning tree of a set of "
    "input points using the dual-tree Boruvka algorithm."
    "\n\n"
    "The set to calculate the minimum spanning tree of is specified with the " +
    PRINT_PARAM_STRING("input") + " parameter, and the output may be saved with"
    " the " + PRINT_PARAM_STRING("output") + " output parameter."
    "\n\n"
    "The " + PRINT_PARAM_STRING("leaf_size") + " parameter controls the leaf "
    "size of the kd-tree that is used to calculate the minimum spanning tree, "
    "and if the " + PRINT_PARAM_STRING("naive") + " option is given, then "
    "brute-force search is used (this is typically much slower in low "
    "dimensions).  The leaf size does not affect the results, but it may have "
    "some effect on the runtime of the algorithm.");

// Example.
BINDING_EXAMPLE(
    "For example, the minimum spanning tree of the input dataset " +
    PRINT_DATASET("data") + " can be calculated with a leaf size of 20 and "
    "stored as " + PRINT_DATASET("spanning_tree") + " using the following "
    "command:"
    "\n\n" +
    PRINT_CALL("emst", "input", "data", "leaf_size", 20, "output",
        "spanning_tree") +
    "\n\n"
    "The output matrix is a three-dimensional matrix, where each row indicates "
    "an edge.  The first dimension corresponds to the lesser index of the edge;"
    " the second dimension corresponds to the greater index of the edge; and "
    "the third column corresponds to the distance between the two points.");

// See also...
BINDING_SEE_ALSO("Minimum spanning tree on Wikipedia",
        "https://en.wikipedia.org/wiki/Minimum_spanning_tree");
BINDING_SEE_ALSO("Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,"
        " and Applications (pdf)", "https://www.mlpack.org/papers/emst.pdf");
BINDING_SEE_ALSO("DualTreeBoruvka class documentation",
        "@src/mlpack/methods/emst/dtb.hpp");

PARAM_MATRIX_IN_REQ("input", "Input data matrix.", "i");
PARAM_MATRIX_OUT("output", "Output data.  Stored as an edge list.", "o");
PARAM_FLAG("naive", "Compute the MST using O(n^2) naive algorithm.", "n");
PARAM_INT_IN("leaf_size", "Leaf size in the kd-tree.  One-element leaves give "
    "the empirically best performance, but at the cost of greater memory "
    "requirements.", "l", 1);

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  arma::mat dataPoints = std::move(params.Get<arma::mat>("input"));

  // Do naive computation if necessary.
  if (params.Get<bool>("naive"))
  {
    Log::Info << "Running naive algorithm." << endl;

    DualTreeBoruvka<> naive(dataPoints, true);

    arma::mat naiveResults;
    naive.ComputeMST(naiveResults);

    if (params.Has("output"))
      params.Get<arma::mat>("output") = std::move(naiveResults);
  }
  else
  {
    Log::Info << "Building tree.\n";

    // Check that the leaf size is reasonable.
    RequireParamValue<int>(params, "leaf_size", [](int x) { return x > 0; },
        true, "leaf size must be greater than or equal to 1");

    // Initialize the tree and get ready to compute the MST.  Compute the tree
    // by hand.
    const size_t leafSize = (size_t) params.Get<int>("leaf_size");

    timers.Start("tree_building");
    std::vector<size_t> oldFromNew;
    KDTree<EuclideanDistance, DTBStat, arma::mat> tree(dataPoints, oldFromNew,
        leafSize);
    LMetric<2, true> metric;
    timers.Stop("tree_building");

    DualTreeBoruvka<> dtb(&tree, metric);

    // Run the DTB algorithm.
    Log::Info << "Calculating minimum spanning tree." << endl;
    arma::mat results;
    timers.Start("mst_computation");
    dtb.ComputeMST(results);
    timers.Stop("mst_computation");

    // Unmap the results.
    arma::mat unmappedResults(results.n_rows, results.n_cols);
    for (size_t i = 0; i < results.n_cols; ++i)
    {
      const size_t indexA = oldFromNew[size_t(results(0, i))];
      const size_t indexB = oldFromNew[size_t(results(1, i))];

      if (indexA < indexB)
      {
        unmappedResults(0, i) = indexA;
        unmappedResults(1, i) = indexB;
      }
      else
      {
        unmappedResults(0, i) = indexB;
        unmappedResults(1, i) = indexA;
      }

      unmappedResults(2, i) = results(2, i);
    }

    if (params.Has("output"))
      params.Get<arma::mat>("output") = std::move(unmappedResults);
  }
}
