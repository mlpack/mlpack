/**
 * @file allknn_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of the AllkNN executable.  Allows some number of standard
 * options.
 *
 * This file is part of MLPACK 1.0.6.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include <string>
#include <fstream>
#include <iostream>

#include "neighbor_search.hpp"
#include "unmap.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;

// Information about the program itself.
PROGRAM_INFO("All K-Nearest-Neighbors",
    "This program will calculate the all k-nearest-neighbors of a set of "
    "points using kd-trees or cover trees (cover tree support is experimental "
    "and may not be optimally fast). You may specify a separate set of "
    "reference points and query points, or just a reference set which will be "
    "used as both the reference and query set."
    "\n\n"
    "For example, the following will calculate the 5 nearest neighbors of each"
    "point in 'input.csv' and store the distances in 'distances.csv' and the "
    "neighbors in the file 'neighbors.csv':"
    "\n\n"
    "$ allknn --k=5 --reference_file=input.csv --distances_file=distances.csv\n"
    "  --neighbors_file=neighbors.csv"
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th nearest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
    "r");
PARAM_STRING_REQ("distances_file", "File to output distances into.", "d");
PARAM_STRING_REQ("neighbors_file", "File to output neighbors into.", "n");

PARAM_INT_REQ("k", "Number of nearest neighbors to find.", "k");

PARAM_STRING("query_file", "File containing query points (optional).", "q", "");

PARAM_INT("leaf_size", "Leaf size for tree building.", "l", 20);
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
    "dual-tree search).", "S");
PARAM_FLAG("cover_tree", "If true, use cover trees to perform the search "
    "(experimental, may be slow).", "c");
PARAM_FLAG("random_basis", "Before tree-building, project the data onto a "
    "random orthogonal basis.", "R");
PARAM_INT("seed", "Random seed (if 0, std::time(NULL) is used).", "s", 0);

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // Get all the parameters.
  const string referenceFile = CLI::GetParam<string>("reference_file");
  const string queryFile = CLI::GetParam<string>("query_file");

  const string distancesFile = CLI::GetParam<string>("distances_file");
  const string neighborsFile = CLI::GetParam<string>("neighbors_file");

  int lsInt = CLI::GetParam<int>("leaf_size");

  size_t k = CLI::GetParam<int>("k");

  bool naive = CLI::HasParam("naive");
  bool singleMode = CLI::HasParam("single_mode");
  const bool randomBasis = CLI::HasParam("random_basis");

  arma::mat referenceData;
  arma::mat queryData; // So it doesn't go out of scope.
  data::Load(referenceFile, referenceData, true);

  Log::Info << "Loaded reference data from '" << referenceFile << "' ("
      << referenceData.n_rows << " x " << referenceData.n_cols << ")." << endl;

  if (queryFile != "")
  {
    data::Load(queryFile, queryData, true);
    Log::Info << "Loaded query data from '" << queryFile << "' ("
      << queryData.n_rows << " x " << queryData.n_cols << ")." << endl;
  }

  // Sanity check on k value: must be greater than 0, must be less than the
  // number of reference points.
  if (k > referenceData.n_cols)
  {
    Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less ";
    Log::Fatal << "than or equal to the number of reference points (";
    Log::Fatal << referenceData.n_cols << ")." << endl;
  }

  // Sanity check on leaf size.
  if (lsInt < 0)
  {
    Log::Fatal << "Invalid leaf size: " << lsInt << ".  Must be greater "
        "than or equal to 0." << endl;
  }
  size_t leafSize = lsInt;

  // Naive mode overrides single mode.
  if (singleMode && naive)
  {
    Log::Warn << "--single_mode ignored because --naive is present." << endl;
  }

  if (naive)
    leafSize = referenceData.n_cols;

  // See if we want to project onto a random basis.
  if (randomBasis)
  {
    // Generate the random basis.
    while (true)
    {
      // [Q, R] = qr(randn(d, d));
      // Q = Q * diag(sign(diag(R)));
      arma::mat q, r;
      if (arma::qr(q, r, arma::randn<arma::mat>(referenceData.n_rows,
          referenceData.n_rows)))
      {
        arma::vec rDiag(r.n_rows);
        for (size_t i = 0; i < rDiag.n_elem; ++i)
        {
          if (r(i, i) < 0)
            rDiag(i) = -1;
          else if (r(i, i) > 0)
            rDiag(i) = 1;
          else
            rDiag(i) = 0;
        }

        q *= arma::diagmat(rDiag);

        // Check if the determinant is positive.
        if (arma::det(q) >= 0)
        {
          referenceData = q * referenceData;
          if (queryFile != "")
            queryData = q * queryData;
          break;
        }
      }
    }
  }

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (!CLI::HasParam("cover_tree"))
  {
    // Because we may construct it differently, we need a pointer.
    AllkNN* allknn = NULL;

    // Mappings for when we build the tree.
    std::vector<size_t> oldFromNewRefs;

    // Build trees by hand, so we can save memory: if we pass a tree to
    // NeighborSearch, it does not copy the matrix.
    Log::Info << "Building reference tree..." << endl;
    Timer::Start("tree_building");

    BinarySpaceTree<bound::HRectBound<2>,
        NeighborSearchStat<NearestNeighborSort> >
        refTree(referenceData, oldFromNewRefs, leafSize);
    BinarySpaceTree<bound::HRectBound<2>,
        NeighborSearchStat<NearestNeighborSort> >*
        queryTree = NULL; // Empty for now.

    Timer::Stop("tree_building");

    std::vector<size_t> oldFromNewQueries;

    if (CLI::GetParam<string>("query_file") != "")
    {
      if (naive && leafSize < queryData.n_cols)
        leafSize = queryData.n_cols;

      Log::Info << "Loaded query data from '" << queryFile << "' ("
          << queryData.n_rows << " x " << queryData.n_cols << ")." << endl;

      Log::Info << "Building query tree..." << endl;

      // Build trees by hand, so we can save memory: if we pass a tree to
      // NeighborSearch, it does not copy the matrix.
      if (!singleMode)
      {
        Timer::Start("tree_building");

        queryTree = new BinarySpaceTree<bound::HRectBound<2>,
            NeighborSearchStat<NearestNeighborSort> >(queryData,
            oldFromNewQueries, leafSize);

        Timer::Stop("tree_building");
      }

      allknn = new AllkNN(&refTree, queryTree, referenceData, queryData,
          singleMode);

      Log::Info << "Tree built." << endl;
    }
    else
    {
      allknn = new AllkNN(&refTree, referenceData, singleMode);

      Log::Info << "Trees built." << endl;
    }

    arma::mat distancesOut;
    arma::Mat<size_t> neighborsOut;

    Log::Info << "Computing " << k << " nearest neighbors..." << endl;
    allknn->Search(k, neighborsOut, distancesOut);

    Log::Info << "Neighbors computed." << endl;

    // We have to map back to the original indices from before the tree
    // construction.
    Log::Info << "Re-mapping indices..." << endl;

    // Map the results back to the correct places.
    if ((CLI::GetParam<string>("query_file") != "") && !singleMode)
      Unmap(neighborsOut, distancesOut, oldFromNewRefs, oldFromNewQueries,
          neighbors, distances);
    else if ((CLI::GetParam<string>("query_file") != "") && singleMode)
      Unmap(neighborsOut, distancesOut, oldFromNewRefs, neighbors, distances);
    else
      Unmap(neighborsOut, distancesOut, oldFromNewRefs, oldFromNewRefs,
          neighbors, distances);

    // Clean up.
    if (queryTree)
      delete queryTree;

    delete allknn;
  }
  else // Cover trees.
  {
    // Make sure to notify the user that they are using cover trees.
    Log::Info << "Using cover trees for nearest-neighbor calculation." << endl;

    // Build our reference tree.
    Log::Info << "Building reference tree..." << endl;
    Timer::Start("tree_building");
    CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
        NeighborSearchStat<NearestNeighborSort> > referenceTree(referenceData,
        1.3);
    CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
        NeighborSearchStat<NearestNeighborSort> >* queryTree = NULL;
    Timer::Stop("tree_building");

    NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
        CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
        NeighborSearchStat<NearestNeighborSort> > >* allknn = NULL;

    // See if we have query data.
    if (CLI::HasParam("query_file"))
    {
      // Build query tree.
      if (!singleMode)
      {
        Log::Info << "Building query tree..." << endl;
        Timer::Start("tree_building");
        queryTree = new CoverTree<metric::LMetric<2, true>,
            tree::FirstPointIsRoot, NeighborSearchStat<NearestNeighborSort> >(
            queryData, 1.3);
        Timer::Stop("tree_building");
      }

      allknn = new NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
          NeighborSearchStat<NearestNeighborSort> > >(&referenceTree, queryTree,
          referenceData, queryData, singleMode);
    }
    else
    {
      allknn = new NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
          NeighborSearchStat<NearestNeighborSort> > >(&referenceTree,
          referenceData, singleMode);
    }

    Log::Info << "Computing " << k << " nearest neighbors..." << endl;
    allknn->Search(k, neighbors, distances);

    Log::Info << "Neighbors computed." << endl;

    delete allknn;

    if (queryTree)
      delete queryTree;
  }

  // Save output.
  data::Save(distancesFile, distances);
  data::Save(neighborsFile, neighbors);
}
