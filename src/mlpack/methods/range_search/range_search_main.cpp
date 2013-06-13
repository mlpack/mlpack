/**
 * @file range_search_main.cpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of the RangeSearch executable.  Allows some number of standard
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
#include <mlpack/core/metrics/lmetric.hpp>

#include "range_search.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::tree;

// Information about the program itself.
PROGRAM_INFO("Range Search",
    "This program implements range search with a Euclidean distance metric. "
    "For a given query point, a given range, and a given set of reference "
    "points, the program will return all of the reference points with distance "
    "to the query point in the given range.  This is performed for an entire "
    "set of query points. You may specify a separate set of reference and query"
    " points, or only a reference set -- which is then used as both the "
    "reference and query set.  The given range is taken to be inclusive (that "
    "is, points with a distance exactly equal to the minimum and maximum of the"
    " range are included in the results)."
    "\n\n"
    "For example, the following will calculate the points within the range [2, "
    "5] of each point in 'input.csv' and store the distances in 'distances.csv'"
    " and the neighbors in 'neighbors.csv':"
    "\n\n"
    "$ range_search --min=2 --max=5 --reference_file=input.csv\n"
    "  --distances_file=distances.csv --neighbors_file=neighbors.csv"
    "\n\n"
    "The output files are organized such that line i corresponds to the points "
    "found for query point i.  Because sometimes 0 points may be found in the "
    "given range, lines of the output files may be empty.  The points are not "
    "ordered in any specific manner."
    "\n\n"
    "Because the number of points returned for each query point may differ, the"
    " resultant CSV-like files may not be loadable by many programs.  However, "
    "at this time a better way to store this non-square result is not known.  "
    "As a result, any output files will be written as CSVs in this manner, "
    "regardless of the given extension.");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
    "r");
PARAM_STRING_REQ("distances_file", "File to output distances into.", "d");
PARAM_STRING_REQ("neighbors_file", "File to output neighbors into.", "n");

PARAM_DOUBLE_REQ("max", "Upper bound in range.", "M");
PARAM_DOUBLE("min", "Lower bound in range.", "m", 0.0);

PARAM_STRING("query_file", "File containing query points (optional).", "q", "");

PARAM_INT("leaf_size", "Leaf size for tree building.", "l", 20);
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
    "dual-tree search.", "s");

typedef RangeSearch<metric::SquaredEuclideanDistance,
    BinarySpaceTree<bound::HRectBound<2>, EmptyStatistic> > RSType;

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  // Get all the parameters.
  string referenceFile = CLI::GetParam<string>("reference_file");

  string distancesFile = CLI::GetParam<string>("distances_file");
  string neighborsFile = CLI::GetParam<string>("neighbors_file");

  int lsInt = CLI::GetParam<int>("leaf_size");

  double max = CLI::GetParam<double>("max");
  double min = CLI::GetParam<double>("min");

  bool naive = CLI::HasParam("naive");
  bool singleMode = CLI::HasParam("single_mode");

  arma::mat referenceData;
  arma::mat queryData; // So it doesn't go out of scope.
  if (!data::Load(referenceFile.c_str(), referenceData))
    Log::Fatal << "Reference file " << referenceFile << "not found." << endl;

  Log::Info << "Loaded reference data from '" << referenceFile << "'." << endl;

  // Sanity check on range value: max must be greater than min.
  if (max <= min)
  {
    Log::Fatal << "Invalid range: maximum (" << max << ") must be greater than "
        << "minimum (" << min << ")." << endl;
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

  vector<vector<size_t> > neighbors;
  vector<vector<double> > distances;

  // Because we may construct it differently, we need a pointer.
  RSType* rangeSearch = NULL;

  // Mappings for when we build the tree.
  vector<size_t> oldFromNewRefs;

  // Build trees by hand, so we can save memory: if we pass a tree to
  // NeighborSearch, it does not copy the matrix.
  Log::Info << "Building reference tree..." << endl;
  Timer::Start("tree_building");

  BinarySpaceTree<bound::HRectBound<2>, tree::EmptyStatistic>
      refTree(referenceData, oldFromNewRefs, leafSize);
  BinarySpaceTree<bound::HRectBound<2>, tree::EmptyStatistic>*
      queryTree = NULL; // Empty for now.

  Timer::Stop("tree_building");

  std::vector<size_t> oldFromNewQueries;

  if (CLI::GetParam<string>("query_file") != "")
  {
    string queryFile = CLI::GetParam<string>("query_file");

    if (!data::Load(queryFile.c_str(), queryData))
      Log::Fatal << "Query file " << queryFile << " not found" << endl;

    if (naive && leafSize < queryData.n_cols)
      leafSize = queryData.n_cols;

    Log::Info << "Loaded query data from '" << queryFile << "'." << endl;

    Log::Info << "Building query tree..." << endl;

    // Build trees by hand, so we can save memory: if we pass a tree to
    // NeighborSearch, it does not copy the matrix.
    Timer::Start("tree_building");

    queryTree = new BinarySpaceTree<bound::HRectBound<2>,
        tree::EmptyStatistic >(queryData, oldFromNewQueries,
        leafSize);

    Timer::Stop("tree_building");

    rangeSearch = new RSType(&refTree, queryTree, referenceData, queryData,
        singleMode);

    Log::Info << "Tree built." << endl;
  }
  else
  {
    rangeSearch = new RSType(&refTree, referenceData, singleMode);

    Log::Info << "Trees built." << endl;
  }

  Log::Info << "Computing neighbors within range [" << min << ", " << max
      << "]." << endl;

  math::Range r = math::Range(min, max);
  rangeSearch->Search(r, neighbors, distances);

  Log::Info << "Neighbors computed." << endl;

  // We have to map back to the original indices from before the tree
  // construction.
  Log::Info << "Re-mapping indices..." << endl;

  vector<vector<double> > distancesOut;
  distancesOut.resize(distances.size());
  vector<vector<size_t> > neighborsOut;
  neighborsOut.resize(neighbors.size());

  // Do the actual remapping.
  if (CLI::GetParam<string>("query_file") != "")
  {
    for (size_t i = 0; i < distances.size(); ++i)
    {
      // Map distances (copy a column).
      distancesOut[oldFromNewQueries[i]] = distances[i];

      // Map indices of neighbors.
      neighborsOut[oldFromNewQueries[i]].resize(neighbors[i].size());
      for (size_t j = 0; j < distances[i].size(); ++j)
      {
        neighborsOut[oldFromNewQueries[i]][j] = oldFromNewRefs[neighbors[i][j]];
      }
    }
  }
  else
  {
    for (size_t i = 0; i < distances.size(); ++i)
    {
      // Map distances (copy a column).
      distancesOut[oldFromNewRefs[i]] = distances[i];

      // Map indices of neighbors.
      neighborsOut[oldFromNewRefs[i]].resize(neighbors[i].size());
      for (size_t j = 0; j < distances[i].size(); ++j)
      {
        neighborsOut[oldFromNewRefs[i]][j] = oldFromNewRefs[neighbors[i][j]];
      }
    }
  }

  // Clean up.
  if (queryTree)
    delete queryTree;

  // Save output.  We have to do this by hand.
  fstream distancesStr(distancesFile.c_str(), fstream::out);
  if (!distancesStr.is_open())
  {
    Log::Warn << "Cannot open file '" << distancesFile << "' to save output "
        << "distances to!" << endl;
  }
  else
  {
    // Loop over each point.
    for (size_t i = 0; i < distancesOut.size(); ++i)
    {
      // Store the distances of each point.  We may have 0 points to store, so
      // we must account for that possibility.
      for (size_t j = 0; j + 1 < distancesOut[i].size(); ++j)
      {
        distancesStr << distancesOut[i][j] << ", ";
      }

      if (distancesOut[i].size() > 0)
        distancesStr << distancesOut[i][distancesOut[i].size() - 1];

      distancesStr << endl;
    }

    distancesStr.close();
  }

  fstream neighborsStr(neighborsFile.c_str(), fstream::out);
  if (!neighborsStr.is_open())
  {
    Log::Warn << "Cannot open file '" << neighborsFile << "' to save output "
        << "neighbor indices to!" << endl;
  }
  else
  {
    // Loop over each point.
    for (size_t i = 0; i < neighborsOut.size(); ++i)
    {
      // Store the neighbors of each point.  We may have 0 points to store, so
      // we must account for that possibility.
      for (size_t j = 0; j + 1 < neighborsOut[i].size(); ++j)
      {
        neighborsStr << neighborsOut[i][j] << ", ";
      }

      if (neighborsOut[i].size() > 0)
        neighborsStr << neighborsOut[i][neighborsOut[i].size() - 1];

      neighborsStr << endl;
    }

    neighborsStr.close();
  }

  delete rangeSearch;
}
