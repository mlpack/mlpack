/**
 * @file methods/range_search/range_search_main.cpp
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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include "range_search.hpp"
#include "rs_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

// Program Name.
BINDING_NAME("Range Search");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of range search with single-tree and dual-tree "
    "algorithms.  Given a set of reference points and a set of query points and"
    " a range, this can find the set of reference points within the desired "
    "range for each query point, and any trees built during the computation can"
    " be saved for reuse with future range searches.");

// Long description.
BINDING_LONG_DESC(
    "This program implements range search with a Euclidean distance metric. "
    "For a given query point, a given range, and a given set of reference "
    "points, the program will return all of the reference points with distance "
    "to the query point in the given range.  This is performed for an entire "
    "set of query points. You may specify a separate set of reference and query"
    " points, or only a reference set -- which is then used as both the "
    "reference and query set.  The given range is taken to be inclusive (that "
    "is, points with a distance exactly equal to the minimum and maximum of the"
    " range are included in the results).");

// Example.
BINDING_EXAMPLE(
    "For example, the following will calculate the points within the range [2, "
    "5] of each point in "+ PRINT_DATASET("input") + " and store the"
    " distances in" + PRINT_DATASET("distances") + " and the neighbors in "
    + PRINT_DATASET("neighbors") +
    "\n\n" +
    PRINT_CALL("range_search", "min", 2, "max", 5, "distances_file", "input",
    "distances_file", "distances", "neighbors_file", "neighbors") +
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

// See also...
BINDING_SEE_ALSO("@knn", "#knn");
BINDING_SEE_ALSO("Range searching on Wikipedia",
        "https://en.wikipedia.org/wiki/Range_searching");
BINDING_SEE_ALSO("Tree-independent dual-tree algorithms (pdf)",
        "http://proceedings.mlr.press/v28/curtin13.pdf");
BINDING_SEE_ALSO("mlpack::range::RangeSearch C++ class documentation",
        "@doxygen/classmlpack_1_1range_1_1RangeSearch.html");

// Define our input parameters that this program will take.
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_STRING_OUT("distances_file", "File to output distances into.", "d");
PARAM_STRING_OUT("neighbors_file", "File to output neighbors into.", "n");

// The option exists to load or save models.
PARAM_MODEL_IN(RSModel, "input_model", "File containing pre-trained range "
    "search model.", "m");
PARAM_MODEL_OUT(RSModel, "output_model", "If specified, the range search model "
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

static void mlpackMain()
{
  if (IO::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) IO::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // A user cannot specify both reference data and a model.
  RequireOnlyOnePassed({ "reference", "input_model" }, true);

  ReportIgnoredParam({{ "input_model", true }}, "tree_type");
  ReportIgnoredParam({{ "input_model", true }}, "random_basis");
  ReportIgnoredParam({{ "input_model", true }}, "leaf_size");
  ReportIgnoredParam({{ "input_model", true }}, "naive");

  // The user must give something to do...
  RequireAtLeastOnePassed({ "min", "max", "output_model" }, false, "no results "
      "will be saved");

  // If the user specifies a range but not output files, they should be warned.
  if (IO::HasParam("min") || IO::HasParam("max"))
  {
    RequireAtLeastOnePassed({ "neighbors_file", "distances_file" }, false,
        "no range search results will be saved");
  }

  if (!IO::HasParam("min") && !IO::HasParam("max"))
  {
    ReportIgnoredParam("neighbors_file", "no range is specified for searching");
    ReportIgnoredParam("distances_file", "no range is specified for searching");
  }

  if (IO::HasParam("input_model") &&
      (IO::HasParam("min") || IO::HasParam("max")))
  {
    RequireAtLeastOnePassed({ "query" }, true, "query set must be passed if "
        "searching is to be done");
  }

  // Sanity check on leaf size.
  int lsInt = IO::GetParam<int>("leaf_size");
  RequireParamValue<int>("leaf_size", [](int x) { return x > 0; }, true,
      "leaf size must be greater than 0");

  // We either have to load the reference data, or we have to load the model.
  RSModel* rs;
  const bool naive = IO::HasParam("naive");
  const bool singleMode = IO::HasParam("single_mode");
  if (IO::HasParam("reference"))
  {
    // Get all the parameters.
    const string treeType = IO::GetParam<string>("tree_type");
    RequireParamInSet<string>("tree_type", { "kd", "cover", "r", "r-star",
        "ball", "x", "hilbert-r", "r-plus", "r-plus-plus", "vp", "rp", "max-rp",
        "ub", "oct" }, true, "unknown tree type");
    const bool randomBasis = IO::HasParam("random_basis");

    rs = new RSModel();

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

    rs->TreeType() = tree;
    rs->RandomBasis() = randomBasis;

    Log::Info << "Using reference data from "
        << IO::GetPrintableParam<arma::mat>("reference") << "." << endl;

    arma::mat referenceSet = std::move(IO::GetParam<arma::mat>("reference"));

    const size_t leafSize = size_t(lsInt);

    rs->BuildModel(std::move(referenceSet), leafSize, naive, singleMode);
  }
  else
  {
    // Load the model from file.
    rs = IO::GetParam<RSModel*>("input_model");

    Log::Info << "Using range search model from '"
        << IO::GetPrintableParam<RSModel*>("input_model") << "' ("
        << "trained on " << rs->Dataset().n_rows << "x" << rs->Dataset().n_cols
        << " dataset)." << endl;

    // Adjust singleMode and naive if necessary.
    rs->SingleMode() = IO::HasParam("single_mode");
    rs->Naive() = IO::HasParam("naive");
    rs->LeafSize() = size_t(lsInt);
  }

  // Perform search, if desired.
  if (IO::HasParam("min") || IO::HasParam("max"))
  {
    const double min = IO::GetParam<double>("min");
    const double max = IO::HasParam("max") ? IO::GetParam<double>("max") :
        DBL_MAX;

    math::Range r(min, max);

    arma::mat queryData;
    if (IO::HasParam("query"))
    {
      Log::Info << "Using query data from "
          << IO::GetPrintableParam<arma::mat>("query") << "." << endl;
      queryData = std::move(IO::GetParam<arma::mat>("query"));
    }

    // Naive mode overrides single mode.
    if (singleMode && naive)
      Log::Warn << PRINT_PARAM_STRING("single_mode") << " ignored because "
          << PRINT_PARAM_STRING("naive") << " is present." << endl;

    // Now run the search.
    vector<vector<size_t>> neighbors;
    vector<vector<double>> distances;

    if (IO::HasParam("query"))
      rs->Search(std::move(queryData), r, neighbors, distances);
    else
      rs->Search(r, neighbors, distances);

    Log::Info << "Search complete." << endl;

    // Save output, if desired.  We have to do this by hand.
    if (IO::HasParam("distances_file"))
    {
      const string distancesFile = IO::GetParam<string>("distances_file");
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

    if (IO::HasParam("neighbors_file"))
    {
      const string neighborsFile = IO::GetParam<string>("neighbors_file");
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

  // Save the output model.
  IO::GetParam<RSModel*>("output_model") = rs;
}
