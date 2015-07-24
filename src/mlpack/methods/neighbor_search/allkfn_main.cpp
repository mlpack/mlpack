/**
 * @file allkfn_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of the AllkFN executable.  Allows some number of standard
 * options.
 */
#include <mlpack/core.hpp>

#include <string>
#include <fstream>
#include <iostream>

#include "neighbor_search.hpp"
#include "unmap.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;

// Information about the program itself.
PROGRAM_INFO("All K-Furthest-Neighbors",
    "This program will calculate the all k-furthest-neighbors of a set of "
    "points. You may specify a separate set of reference points and query "
    "points, or just a reference set which will be used as both the reference "
    "and query set."
    "\n\n"
    "For example, the following will calculate the 5 furthest neighbors of each"
    "point in 'input.csv' and store the distances in 'distances.csv' and the "
    "neighbors in the file 'neighbors.csv':"
    "\n\n"
    "$ allkfn --k=5 --reference_file=input.csv --distances_file=distances.csv\n"
    "  --neighbors_file=neighbors.csv"
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th furthest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
    "r");
PARAM_INT_REQ("k", "Number of furthest neighbors to find.", "k");
PARAM_STRING_REQ("distances_file", "File to output distances into.", "d");
PARAM_STRING_REQ("neighbors_file", "File to output neighbors into.", "n");

PARAM_STRING("query_file", "File containing query points (optional).", "q", "");

PARAM_INT("leaf_size", "Leaf size for tree building.", "l", 20);
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
    "dual-tree search).", "s");
PARAM_FLAG("r_tree", "If true, use an R-Tree to perform the search "
    "(experimental, may be slow.).", "T");

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  // Get all the parameters.
  string referenceFile = CLI::GetParam<string>("reference_file");

  string distancesFile = CLI::GetParam<string>("distances_file");
  string neighborsFile = CLI::GetParam<string>("neighbors_file");

  int lsInt = CLI::GetParam<int>("leaf_size");

  size_t k = CLI::GetParam<int>("k");

  bool naive = CLI::HasParam("naive");
  bool singleMode = CLI::HasParam("single_mode");

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

  if (CLI::GetParam<string>("query_file") != "")
  {
    string queryFile = CLI::GetParam<string>("query_file");
    data::Load(queryFile, queryData, true);
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

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (naive)
  {
    AllkFN allkfn(referenceData, false, naive);

    if (CLI::HasParam("query_file"))
      allkfn.Search(queryData, k, neighbors, distances);
    else
      allkfn.Search(k, neighbors, distances);
  }
  if (!CLI::HasParam("r_tree"))
  {
    // Use default kd-tree.
    std::vector<size_t> oldFromNewRefs;

    typedef KDTree<EuclideanDistance, NeighborSearchStat<FurthestNeighborSort>,
        arma::mat> TreeType;

    // Build trees by hand, so we can save memory: if we pass a tree to
    // NeighborSearch, it does not copy the matrix.
    Log::Info << "Building reference tree..." << endl;
    Timer::Start("reference_tree_building");
    TreeType refTree(referenceData, oldFromNewRefs, leafSize);
    Timer::Stop("reference_tree_building");

    std::vector<size_t> oldFromNewQueries;

    AllkFN allkfn(&refTree, singleMode);

    arma::mat distancesOut(distances.n_rows, distances.n_cols);
    arma::Mat<size_t> neighborsOut(neighbors.n_rows, neighbors.n_cols);

    if (CLI::HasParam("query_file"))
    {
      if (!singleMode)
      {
        // Build trees by hand, so we can save memory: if we pass a tree to
        // NeighborSearch, it does not copy the matrix.
        Log::Info << "Building query tree..." << endl;
        Timer::Start("tree_building");
        TreeType queryTree(queryData, oldFromNewQueries, leafSize);
        Timer::Stop("tree_building");
        Log::Info << "Tree built." << endl;

        Log::Info << "Computing " << k << " furthest neighbors..." << endl;
        allkfn.Search(&queryTree, k, neighborsOut, distancesOut);
      }
      else
      {
        Log::Info << "Computing " << k << " furthest neighbors..." << endl;
        allkfn.Search(queryData, k, neighborsOut, distancesOut);
      }
    }
    else
    {
      Log::Info << "Computing " << k << " furthest neighbors..." << endl;
      allkfn.Search(k, neighborsOut, distancesOut);
    }

    Log::Info << "Neighbors computed." << endl;

    // We have to map back to the original indices from before the tree
    // construction.
    Log::Info << "Re-mapping indices..." << endl;

    // Map the points back to their original locations.
    if ((CLI::GetParam<string>("query_file") != "") && !singleMode)
      Unmap(neighborsOut, distancesOut, oldFromNewRefs, oldFromNewQueries, neighbors,
          distances);
    else if ((CLI::GetParam<string>("query_file") != "") && singleMode)
      Unmap(neighborsOut, distancesOut, oldFromNewRefs, neighbors, distances);
    else
      Unmap(neighborsOut, distancesOut, oldFromNewRefs, oldFromNewRefs, neighbors,
          distances);
  }
  else
  {
    // Use the R tree.
    Log::Info << "Using R tree for furthest-neighbor calculation." << endl;

    // Convenience typedef.
    typedef RStarTree<EuclideanDistance,
        NeighborSearchStat<FurthestNeighborSort>, arma::mat> TreeType;

    // Build trees by hand, so we can save memory: if we pass a tree to
    // NeighborSearch, it does not copy the matrix.
    Log::Info << "Building reference tree..." << endl;
    Timer::Start("tree_building");
    TreeType refTree(referenceData, leafSize, leafSize * 0.4, 5, 2, 0);
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << endl;

    typedef NeighborSearch<FurthestNeighborSort, EuclideanDistance, arma::mat,
        RStarTree> AllkFNType;
    AllkFNType allkfn(&refTree, singleMode);

    if (CLI::GetParam<string>("query_file") != "")
    {
      if (!singleMode)
      {
        Timer::Start("tree_building");
        TreeType queryTree(queryData, leafSize, leafSize * 0.4, 5, 2, 0);
        Timer::Stop("tree_building");

        Log::Info << "Computing " << k << " nearest neighbors..." << endl;
        allkfn.Search(&queryTree, k, neighbors, distances);
      }
      else
      {
        Log::Info << "Computing " << k << " nearest neighbors..." << endl;
        allkfn.Search(queryData, k, neighbors, distances);
      }
    }
    else
    {
      Log::Info << "Computing " << k << " nearest neighbors..." << endl;
      allkfn.Search(k, neighbors, distances);
    }
    Log::Info << "Neighbors computed." << endl;
  }

  // Save output.
  data::Save(distancesFile, distances);
  data::Save(neighborsFile, neighbors);
}
