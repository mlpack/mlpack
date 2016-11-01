/**
 * @file emst_main.cpp
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
#include "dtb.hpp"

#include <mlpack/core.hpp>

PROGRAM_INFO("Fast Euclidean Minimum Spanning Tree", "This program can compute "
    "the Euclidean minimum spanning tree of a set of input points using the "
    "dual-tree Boruvka algorithm."
    "\n\n"
    "The output is saved in a three-column matrix, where each row indicates an "
    "edge.  The first column corresponds to the lesser index of the edge; the "
    "second column corresponds to the greater index of the edge; and the third "
    "column corresponds to the distance between the two points.");

PARAM_MATRIX_IN_REQ("input", "Input data matrix.", "i");
PARAM_MATRIX_OUT("output", "Output data.  Stored as an edge list.", "o");
PARAM_FLAG("naive", "Compute the MST using O(n^2) naive algorithm.", "n");
PARAM_INT_IN("leaf_size", "Leaf size in the kd-tree.  One-element leaves give "
    "the empirically best performance, but at the cost of greater memory "
    "requirements.", "l", 1);

using namespace mlpack;
using namespace mlpack::emst;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace std;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  if (!CLI::HasParam("output"))
    Log::Warn << "--output_file is not specified, so no output will be saved!"
        << endl;

  arma::mat dataPoints = std::move(CLI::GetParam<arma::mat>("input"));

  // Do naive computation if necessary.
  if (CLI::GetParam<bool>("naive"))
  {
    Log::Info << "Running naive algorithm." << endl;

    DualTreeBoruvka<> naive(dataPoints, true);

    arma::mat naiveResults;
    naive.ComputeMST(naiveResults);

    if (CLI::HasParam("output"))
      CLI::GetParam<arma::mat>("output") = std::move(naiveResults);
  }
  else
  {
    Log::Info << "Building tree.\n";

    // Check that the leaf size is reasonable.
    if (CLI::GetParam<int>("leaf_size") <= 0)
    {
      Log::Fatal << "Invalid leaf size (" << CLI::GetParam<int>("leaf_size")
          << ")!  Must be greater than or equal to 1." << std::endl;
    }

    // Initialize the tree and get ready to compute the MST.  Compute the tree
    // by hand.
    const size_t leafSize = (size_t) CLI::GetParam<int>("leaf_size");

    Timer::Start("tree_building");
    std::vector<size_t> oldFromNew;
    KDTree<EuclideanDistance, DTBStat, arma::mat> tree(dataPoints, oldFromNew,
        leafSize);
    metric::LMetric<2, true> metric;
    Timer::Stop("tree_building");

    DualTreeBoruvka<> dtb(&tree, metric);

    // Run the DTB algorithm.
    Log::Info << "Calculating minimum spanning tree." << endl;
    arma::mat results;
    dtb.ComputeMST(results);

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

    if (CLI::HasParam("output"))
      CLI::GetParam<arma::mat>("output") = std::move(unmappedResults);
  }
}
