/**
 * @file range_search_main.cpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of the RangeSearch executable.  Allows some number of standard
 * options.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include "range_search.hpp"
#include "rs_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::tree;
using namespace mlpack::metric;

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
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_STRING_OUT("distances_file", "File to output distances into.", "d");
PARAM_STRING_OUT("neighbors_file", "File to output neighbors into.", "n");

// The option exists to load or save models.
PARAM_STRING_IN("input_model_file", "File containing pre-trained range search "
    "model.", "m", "");
PARAM_STRING_OUT("output_model_file", "If specified, the range search model "
    "will be saved to the given file.", "M");

// The user may specify a query file of query points and a range to search for.
PARAM_MATRIX_IN("query", "File containing query points (optional).", "q");
PARAM_DOUBLE_IN("max", "Upper bound in range (if not specified, +inf will be "
    "used.", "U", 0.0);
PARAM_DOUBLE_IN("min", "Lower bound in range.", "L", 0.0);

// The user may specify the type of tree to use, and a few parameters for tree
// building.
PARAM_STRING_IN("tree_type", "Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', "
    "'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', "
    "'r-plus-plus', 'oct'.", "t", "kd");
PARAM_INT_IN("leaf_size", "Leaf size for tree building (used for kd-trees, "
    "vp trees, random projection trees, UB trees, R trees, R* trees, X trees, "
    "Hilbert R trees, R+ trees, R++ trees, and octrees).", "l", 20);
PARAM_FLAG("random_basis", "Before tree-building, project the data onto a "
    "random orthogonal basis.", "R");
PARAM_INT_IN("seed", "Random seed (if 0, std::time(NULL) is used).", "s", 0);

// Search settings.
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
    "dual-tree search).", "S");

typedef RangeSearch<> RSType;
typedef CoverTree<EuclideanDistance, RangeSearchStat> CoverTreeType;
typedef RangeSearch<EuclideanDistance, arma::mat, StandardCoverTree>
    RSCoverType;

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // A user cannot specify both reference data and a model.
  if (CLI::HasParam("reference") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Only one of --reference_file (-r) or --input_model_file (-m)"
        << " may be specified!" << endl;

  // A user must specify one of them...
  if (!CLI::HasParam("reference") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "No model specified (--input_model_file) and no reference "
        << "data specified (--reference_file)!  One must be provided." << endl;

  if (CLI::HasParam("input_model_file"))
  {
    // Notify the user of parameters that will be ignored.
    if (CLI::HasParam("tree_type"))
      Log::Warn << "--tree_type (-t) will be ignored because --input_model_file"
          << " is specified." << endl;
    if (CLI::HasParam("leaf_size"))
      Log::Warn << "--leaf_size (-l) will be ignored because --input_model_file"
          << " is specified." << endl;
    if (CLI::HasParam("random_basis"))
      Log::Warn << "--random_basis (-R) will be ignored because "
          << "--input_model_file is specified." << endl;
    if (CLI::HasParam("naive"))
      Log::Warn << "--naive (-N) will be ignored because --input_model_file is "
          << "specified." << endl;
  }

  // The user must give something to do...
  if (!CLI::HasParam("min") && !CLI::HasParam("max") &&
      !CLI::HasParam("output_model_file"))
    Log::Warn << "Neither --min, --max, nor --output_model_file are specified, "
        << "so no results from this program will be saved!" << endl;

  // If the user specifies a range but not output files, they should be warned.
  if ((CLI::HasParam("min") || CLI::HasParam("max")) &&
      !(CLI::HasParam("neighbors_file") || CLI::HasParam("distances_file")))
    Log::Warn << "Neither --neighbors_file nor --distances_file is specified, "
        << "so the range search results will not be saved!" << endl;

  // If the user specifies output files but no range, they should be warned.
  if ((CLI::HasParam("neighbors_file") || CLI::HasParam("distances_file")) &&
      !(CLI::HasParam("min") || CLI::HasParam("max")))
    Log::Warn << "An output file for range search is given (--neighbors_file "
        << "or --distances_file), but range search is not being performed "
        << "because neither --min nor --max are specified!  No results will be "
        << "saved." << endl;

  // Sanity check on leaf size.
  int lsInt = CLI::GetParam<int>("leaf_size");
  if (lsInt < 1)
    Log::Fatal << "Invalid leaf size: " << lsInt << ".  Must be greater than 0."
        << endl;

  // We either have to load the reference data, or we have to load the model.
  RSModel rs;
  const bool naive = CLI::HasParam("naive");
  const bool singleMode = CLI::HasParam("single_mode");
  if (CLI::HasParam("reference"))
  {
    // Get all the parameters.
    const string treeType = CLI::GetParam<string>("tree_type");
    const bool randomBasis = CLI::HasParam("random_basis");

    RSModel::TreeTypes tree = RSModel::KD_TREE;
    if (treeType == "kd")
      tree = RSModel::KD_TREE;
    else if (treeType == "cover")
      tree = RSModel::COVER_TREE;
    else if (treeType == "r")
      tree = RSModel::R_TREE;
    else if (treeType == "r-star")
      tree = RSModel::R_STAR_TREE;
    else if (treeType == "ball")
      tree = RSModel::BALL_TREE;
    else if (treeType == "x")
      tree = RSModel::X_TREE;
    else if (treeType == "hilbert-r")
      tree = RSModel::HILBERT_R_TREE;
    else if (treeType == "r-plus")
      tree = RSModel::R_PLUS_TREE;
    else if (treeType == "r-plus-plus")
      tree = RSModel::R_PLUS_PLUS_TREE;
    else if (treeType == "vp")
      tree = RSModel::VP_TREE;
    else if (treeType == "rp")
      tree = RSModel::RP_TREE;
    else if (treeType == "max-rp")
      tree = RSModel::MAX_RP_TREE;
    else if (treeType == "ub")
      tree = RSModel::UB_TREE;
    else if (treeType == "oct")
      tree = RSModel::OCTREE;
    else
      Log::Fatal << "Unknown tree type '" << treeType << "; valid choices are "
          << "'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', "
          << "'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', and 'oct'." << endl;

    rs.TreeType() = tree;
    rs.RandomBasis() = randomBasis;

    arma::mat referenceSet = std::move(CLI::GetParam<arma::mat>("reference"));

    Log::Info << "Loaded reference data from '"
        << CLI::GetUnmappedParam<arma::mat>("reference") << "' ("
        << referenceSet.n_rows << "x" << referenceSet.n_cols << ")." << endl;

    const size_t leafSize = size_t(lsInt);

    rs.BuildModel(std::move(referenceSet), leafSize, naive, singleMode);
  }
  else
  {
    // Load the model from file.
    const string inputModelFile = CLI::GetParam<string>("input_model_file");
    data::Load(inputModelFile, "rs_model", rs, true); // Fatal on failure.

    Log::Info << "Loaded range search model from '" << inputModelFile << "' ("
        << "trained on " << rs.Dataset().n_rows << "x" << rs.Dataset().n_cols
        << " dataset)." << endl;

    // Adjust singleMode and naive if necessary.
    rs.SingleMode() = CLI::HasParam("single_mode");
    rs.Naive() = CLI::HasParam("naive");
    rs.LeafSize() = size_t(lsInt);
  }

  // Perform search, if desired.
  if (CLI::HasParam("min") || CLI::HasParam("max"))
  {
    const double min = CLI::GetParam<double>("min");
    const double max = CLI::HasParam("max") ? CLI::GetParam<double>("max") :
        DBL_MAX;

    math::Range r(min, max);

    arma::mat queryData;
    if (CLI::HasParam("query"))
    {
      queryData = std::move(CLI::GetParam<arma::mat>("query"));
      Log::Info << "Loaded query data from '"
          << CLI::GetUnmappedParam<arma::mat>("query") << "' ("
          << queryData.n_rows << "x" << queryData.n_cols << ")." << endl;
    }

    // Naive mode overrides single mode.
    if (singleMode && naive)
      Log::Warn << "--single_mode ignored because --naive is present." << endl;

    // Now run the search.
    vector<vector<size_t>> neighbors;
    vector<vector<double>> distances;

    if (CLI::HasParam("query"))
      rs.Search(std::move(queryData), r, neighbors, distances);
    else
      rs.Search(r, neighbors, distances);

    Log::Info << "Search complete." << endl;

    // Save output, if desired.  We have to do this by hand.
    if (CLI::HasParam("distances_file"))
    {
      const string distancesFile = CLI::GetParam<string>("distances_file");
      fstream distancesStr(distancesFile.c_str(), fstream::out);
      if (!distancesStr.is_open())
      {
        Log::Warn << "Cannot open file '" << distancesFile << "' to save output"
            << " distances to!" << endl;
      }
      else
      {
        // Loop over each point.
        for (size_t i = 0; i < distances.size(); ++i)
        {
          // Store the distances of each point.  We may have 0 points to store,
          // so we must account for that possibility.
          for (size_t j = 0; j + 1 < distances[i].size(); ++j)
            distancesStr << distances[i][j] << ", ";

          if (distances[i].size() > 0)
            distancesStr << distances[i][distances[i].size() - 1];

          distancesStr << endl;
        }

        distancesStr.close();
      }
    }

    if (CLI::HasParam("neighbors_file"))
    {
      const string neighborsFile = CLI::GetParam<string>("neighbors_file");
      fstream neighborsStr(neighborsFile.c_str(), fstream::out);
      if (!neighborsStr.is_open())
      {
        Log::Warn << "Cannot open file '" << neighborsFile << "' to save output"
            << " neighbor indices to!" << endl;
      }
      else
      {
        // Loop over each point.
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
          // Store the neighbors of each point.  We may have 0 points to store,
          // so we must account for that possibility.
          for (size_t j = 0; j + 1 < neighbors[i].size(); ++j)
            neighborsStr << neighbors[i][j] << ", ";

          if (neighbors[i].size() > 0)
            neighborsStr << neighbors[i][neighbors[i].size() - 1];

          neighborsStr << endl;
        }

        neighborsStr.close();
      }
    }
  }

  // Save the output model, if desired.
  if (CLI::HasParam("output_model_file"))
  {
    const string outputModelFile = CLI::GetParam<string>("output_model_file");
    data::Save(outputModelFile, "rs_model", rs);
  }
}
