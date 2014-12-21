/**
 * @file allkrann_main.cpp
 * @author Parikshit Ram
 *
 * Implementation of the AllkRANN executable.  Allows some number of standard
 * options.
 *
 * This file is part of MLPACK 1.0.9.
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
#include <time.h>

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include <string>
#include <fstream>
#include <iostream>

#include "ra_search.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;

// Information about the program itself.
PROGRAM_INFO("All K-Rank-Approximate-Nearest-Neighbors",
    "This program will calculate the k rank-approximate-nearest-neighbors of a "
    "set of points. You may specify a separate set of reference points and "
    "query points, or just a reference set which will be used as both the "
    "reference and query set. You must specify the rank approximation (in \%) "
    "(and optionally the success probability)."
    "\n\n"
    "For example, the following will return 5 neighbors from the top 0.1\% of "
    "the data (with probability 0.95) for each point in 'input.csv' and store "
    "the distances in 'distances.csv' and the neighbors in the file "
    "'neighbors.csv':"
    "\n\n"
    "$ allkrann -k 5 -r input.csv -d distances.csv -n neighbors.csv --tau 0.1"
    "\n\n"
    "Note that tau must be set such that the number of points in the "
    "corresponding percentile of the data is greater than k.  Thus, if we "
    "choose tau = 0.1 with a dataset of 1000 points and k = 5, then we are "
    "attempting to choose 5 nearest neighbors out of the closest 1 point -- "
    "this is invalid and the program will terminate with an error message."
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th nearest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
                 "r");
PARAM_STRING("distances_file", "File to output distances into.", "d", "");
PARAM_STRING("neighbors_file", "File to output neighbors into.", "n", "");

PARAM_INT_REQ("k", "Number of nearest neighbors to find.", "k");

PARAM_STRING("query_file", "File containing query points (optional).",
             "q", "");

PARAM_DOUBLE("tau", "The allowed rank-error in terms of the percentile of "
             "the data.", "t", 5);
PARAM_DOUBLE("alpha", "The desired success probability.", "a", 0.95);

PARAM_INT("leaf_size", "Leaf size for tree building.", "l", 20);
PARAM_FLAG("naive", "If true, sampling will be done without using a tree.",
           "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
           "dual-tree search.", "s");
PARAM_FLAG("cover_tree", "If true, use cover trees to perform the search.",
           "c");

PARAM_FLAG("sample_at_leaves", "The flag to trigger sampling at leaves.", "L");
PARAM_FLAG("first_leaf_exact", "The flag to trigger sampling only after "
           "exactly exploring the first leaf.", "X");
PARAM_INT("single_sample_limit", "The limit on the maximum number of "
    "samples (and hence the largest node you can approximate).", "S", 20);

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);
  math::RandomSeed(time(NULL));

  // Get all the parameters.
  string referenceFile = CLI::GetParam<string>("reference_file");
  string distancesFile = CLI::GetParam<string>("distances_file");
  string neighborsFile = CLI::GetParam<string>("neighbors_file");

  int lsInt = CLI::GetParam<int>("leaf_size");
  size_t singleSampleLimit = CLI::GetParam<int>("single_sample_limit");

  size_t k = CLI::GetParam<int>("k");

  double tau = CLI::GetParam<double>("tau");
  double alpha = CLI::GetParam<double>("alpha");

  bool naive = CLI::HasParam("naive");
  bool singleMode = CLI::HasParam("single_mode");
  bool sampleAtLeaves = CLI::HasParam("sample_at_leaves");
  bool firstLeafExact = CLI::HasParam("first_leaf_exact");

  arma::mat referenceData;
  arma::mat queryData; // So it doesn't go out of scope.
  data::Load(referenceFile, referenceData, true);

  Log::Info << "Loaded reference data from '" << referenceFile << "' ("
      << referenceData.n_rows << " x " << referenceData.n_cols << ")." << endl;

  // Sanity check on k value: must be greater than 0, must be less than the
  // number of reference points.
  if (k > referenceData.n_cols)
  {
    Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less ";
    Log::Fatal << "than or equal to the number of reference points (";
    Log::Fatal << referenceData.n_cols << ")." << endl;
  }

  // Sanity check on the value of 'tau' with respect to 'k' so that
  // 'k' neighbors are not requested from the top-'rank_error' neighbors
  // where 'rank_error' <= 'k'.
  size_t rank_error = (size_t) ceil(tau *
      (double) referenceData.n_cols / 100.0);
  if (rank_error <= k)
    Log::Fatal << "Invalid 'tau' (" << tau << ") - k (" << k << ") " <<
      "combination. Increase 'tau' or decrease 'k'." << endl;

  // Sanity check on leaf size.
  if (lsInt < 0)
    Log::Fatal << "Invalid leaf size: " << lsInt << ".  Must be greater "
      "than or equal to 0." << endl;
  size_t leafSize = lsInt;

  // Naive mode overrides single mode.
  if (singleMode && naive)
    Log::Warn << "--single_mode ignored because --naive is present." << endl;

  // The actual output after the remapping.
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (naive)
  {
    AllkRANN* allkrann;
    if (CLI::GetParam<string>("query_file") != "")
    {
      string queryFile = CLI::GetParam<string>("query_file");

      data::Load(queryFile, queryData, true);

      Log::Info << "Loaded query data from '" << queryFile << "' (" <<
        queryData.n_rows << " x " << queryData.n_cols << ")." << endl;

      allkrann = new AllkRANN(referenceData, queryData, naive);
    }
    else
      allkrann = new AllkRANN(referenceData, naive);

    Log::Info << "Computing " << k << " nearest neighbors " << "with " <<
      tau << "% rank approximation..." << endl;

    allkrann->Search(k, neighbors, distances, tau, alpha);

    Log::Info << "Neighbors computed." << endl;

    delete allkrann;
  }
  else
  {
    // The results output by the AllkRANN class
    // shuffled because the tree construction shuffles the point sets.
    arma::Mat<size_t> neighborsOut;
    arma::mat distancesOut;

    if (!CLI::HasParam("cover_tree"))
    {
      // Because we may construct it differently, we need a pointer.
      AllkRANN* allkrann = NULL;

      // Mappings for when we build the tree.
      std::vector<size_t> oldFromNewRefs;

      // Build trees by hand, so we can save memory: if we pass a tree to
      // NeighborSearch, it does not copy the matrix.
      Log::Info << "Building reference tree..." << endl;
      Timer::Start("tree_building");

      BinarySpaceTree<bound::HRectBound<2, false>,
          RAQueryStat<NearestNeighborSort> >
          refTree(referenceData, oldFromNewRefs, leafSize);
      BinarySpaceTree<bound::HRectBound<2, false>,
          RAQueryStat<NearestNeighborSort> >*
          queryTree = NULL; // Empty for now.

      Timer::Stop("tree_building");

      std::vector<size_t> oldFromNewQueries;

      if (CLI::GetParam<string>("query_file") != "")
      {
        string queryFile = CLI::GetParam<string>("query_file");

        data::Load(queryFile, queryData, true);

        if (naive && leafSize < queryData.n_cols)
          leafSize = queryData.n_cols;

        Log::Info << "Loaded query data from '" << queryFile << "' (" <<
          queryData.n_rows << " x " << queryData.n_cols << ")." << endl;

        Log::Info << "Building query tree..." << endl;

        // Build trees by hand, so we can save memory: if we pass a tree to
        // NeighborSearch, it does not copy the matrix.
        Timer::Start("tree_building");

        queryTree = new BinarySpaceTree<bound::HRectBound<2, false>,
            RAQueryStat<NearestNeighborSort> >
            (queryData, oldFromNewQueries, leafSize);
        Timer::Stop("tree_building");

        allkrann = new AllkRANN(&refTree, queryTree, referenceData, queryData,
                                singleMode);

        Log::Info << "Tree built." << endl;
      }
      else
      {
        allkrann = new AllkRANN(&refTree, referenceData, singleMode);
        Log::Info << "Trees built." << endl;
      }

      Log::Info << "Computing " << k << " nearest neighbors " << "with " <<
        tau << "% rank approximation..." << endl;
      allkrann->Search(k, neighborsOut, distancesOut,
                       tau, alpha, sampleAtLeaves,
                       firstLeafExact, singleSampleLimit);

      Log::Info << "Neighbors computed." << endl;

      // We have to map back to the original indices from before the tree
      // construction.
      Log::Info << "Re-mapping indices..." << endl;

      neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
      distances.set_size(distancesOut.n_rows, distancesOut.n_cols);

      // Do the actual remapping.
      if (CLI::GetParam<string>("query_file") != "")
      {
        for (size_t i = 0; i < distancesOut.n_cols; ++i)
        {
          // Map distances (copy a column).
          distances.col(oldFromNewQueries[i]) = distancesOut.col(i);

          // Map indices of neighbors.
          for (size_t j = 0; j < distancesOut.n_rows; ++j)
          {
            neighbors(j, oldFromNewQueries[i])
              = oldFromNewRefs[neighborsOut(j, i)];
          }
        }
      }
      else
      {
        for (size_t i = 0; i < distancesOut.n_cols; ++i)
        {
          // Map distances (copy a column).
          distances.col(oldFromNewRefs[i]) = distancesOut.col(i);

          // Map indices of neighbors.
          for (size_t j = 0; j < distancesOut.n_rows; ++j)
          {
            neighbors(j, oldFromNewRefs[i])
              = oldFromNewRefs[neighborsOut(j, i)];
          }
        }
      }

      // Clean up.
      if (queryTree)
        delete queryTree;

      delete allkrann;
    }
    else // Cover trees.
    {
      Log::Fatal << "Cover tree case not implemented yet..." << endl;
    }
  }

  // Save output.
  if (distancesFile != "")
    data::Save(distancesFile, distances);
  if (neighborsFile != "")
    data::Save(neighborsFile, neighbors);
}
