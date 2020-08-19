/**
 * @file methods/neighbor_search/kfn_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of the KFN executable.  Allows some number of standard
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

#include <string>
#include <fstream>
#include <iostream>

#include "neighbor_search.hpp"
#include "unmap.hpp"
#include "ns_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

// Convenience typedef.
typedef NSModel<FurthestNS> KFNModel;

// Program Name.
BINDING_NAME("k-Furthest-Neighbors Search");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of k-furthest-neighbor search using single-tree and "
    "dual-tree algorithms.  Given a set of reference points and query points, "
    "this can find the k furthest neighbors in the reference set of each query"
    " point using trees; trees that are built can be saved for future use.");

// Long description.
BINDING_LONG_DESC(
    "This program will calculate the k-furthest-neighbors of a set of "
    "points. You may specify a separate set of reference points and query "
    "points, or just a reference set which will be used as both the reference "
    "and query set.");

// Example.
BINDING_EXAMPLE(
    "For example, the following will calculate the 5 furthest neighbors of each"
    "point in " + PRINT_DATASET("input") + " and store the distances in " +
    PRINT_DATASET("distances") + " and the neighbors in " +
    PRINT_DATASET("neighbors") + ": "
    "\n\n" +
    PRINT_CALL("kfn", "k", 5, "reference", "input", "distances", "distances",
        "neighbors", "neighbors") +
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output matrix corresponds to the index of the point in the "
    "reference set which is the j'th furthest neighbor from the point in the "
    "query set with index i.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// See also...
BINDING_SEE_ALSO("@approx_kfn", "#approx_kfn");
BINDING_SEE_ALSO("@knn", "#knn");
BINDING_SEE_ALSO("Tree-independent dual-tree algorithms (pdf)",
        "http://proceedings.mlr.press/v28/curtin13.pdf");
BINDING_SEE_ALSO("mlpack::neighbor::NeighborSearch C++ class documentation",
        "@doxygen/classmlpack_1_1neighbor_1_1NeighborSearch.html");

// Define our input parameters that this program will take.
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");
PARAM_MATRIX_IN("true_distances", "Matrix of true distances to compute "
    "the effective error (average relative error) (it is printed when -v is "
    "specified).", "D");
PARAM_UMATRIX_IN("true_neighbors", "Matrix of true neighbors to compute the "
    "recall (it is printed when -v is specified).", "T");

// The option exists to load or save models.
PARAM_MODEL_IN(KFNModel, "input_model", "Pre-trained kFN model.", "m");
PARAM_MODEL_OUT(KFNModel, "output_model", "If specified, the kFN model will be "
    "output here.", "M");

// The user may specify a query file of query points and a number of furthest
// neighbors to search for.
PARAM_MATRIX_IN("query", "Matrix containing query points (optional).", "q");
PARAM_INT_IN("k", "Number of furthest neighbors to find.", "k", 0);

// The user may specify the type of tree to use, and a few pararmeters for tree
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
PARAM_STRING_IN("algorithm", "Type of neighbor search: 'naive', 'single_tree', "
    "'dual_tree', 'greedy'.", "a", "dual_tree");
PARAM_DOUBLE_IN("epsilon", "If specified, will do approximate furthest neighbor"
    " search with given relative error. Must be in the range [0,1).", "e", 0);
PARAM_DOUBLE_IN("percentage", "If specified, will do approximate furthest "
    "neighbor search. Must be in the range (0,1] (decimal form). Resultant "
    "neighbors will be at least (p*100) % of the distance as the true furthest "
    "neighbor.", "p", 1);

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

  // Notify the user of parameters that will be only be considered for query
  // tree.
  if (IO::HasParam("input_model") && IO::HasParam("leaf_size"))
  {
    Log::Warn << PRINT_PARAM_STRING("leaf_size") << " will only be considered"
        << " for the query tree, because "
        << PRINT_PARAM_STRING("input_model") << " is specified." << endl;
  }

  // The user should give something to do...
  RequireAtLeastOnePassed({ "k", "output_model" }, false,
      "no results will be saved");

  // If the user specifies k but no output files, they should be warned.
  if (IO::HasParam("k"))
  {
    RequireAtLeastOnePassed({ "neighbors", "distances" }, false,
        "furthest neighbor search results will not be saved");
  }

  // If the user specifies output files but no k, they should be warned.
  ReportIgnoredParam({{ "k", false }}, "neighbors");
  ReportIgnoredParam({{ "k", false }}, "distances");
  ReportIgnoredParam({{ "k", false }}, "true_neighbors");
  ReportIgnoredParam({{ "k", false }}, "true_distances");
  ReportIgnoredParam({{ "k", false }}, "query");

  // Sanity check on leaf size.
  RequireParamValue<int>("leaf_size", [](int x) { return x > 0; },
      true, "leaf size must be positive");
  const int lsInt = IO::GetParam<int>("leaf_size");

  // Sanity check on epsilon.
  double epsilon = IO::GetParam<double>("epsilon");
  RequireParamValue<double>("epsilon", [](double x)
      { return x >= 0.0 && x < 1; }, true,
          "epsilon must be in the range [0, 1).");

  // Sanity check on percentage.
  const double percentage = IO::GetParam<double>("percentage");
  RequireParamValue<double>("percentage",
      [](double x) { return x > 0.0 && x <= 1.0; }, true,
      "percentage must be in the range (0, 1]");

  ReportIgnoredParam({{ "epsilon", true }}, "percentage");

  if (IO::HasParam("percentage"))
    epsilon = 1 - percentage;

  // We either have to load the reference data, or we have to load the model.
  NSModel<FurthestNS>* kfn;

  const string algorithm = IO::GetParam<string>("algorithm");
  RequireParamInSet<string>("algorithm", { "naive", "single_tree", "dual_tree",
      "greedy" }, true, "unknown neighbor search algorithm");
  NeighborSearchMode searchMode = DUAL_TREE_MODE;

  if (algorithm == "naive")
    searchMode = NAIVE_MODE;
  else if (algorithm == "single_tree")
    searchMode = SINGLE_TREE_MODE;
  else if (algorithm == "dual_tree")
    searchMode = DUAL_TREE_MODE;
  else if (algorithm == "greedy")
    searchMode = GREEDY_SINGLE_TREE_MODE;

  if (IO::HasParam("reference"))
  {
    // Get all the parameters.
    RequireParamInSet<string>("tree_type", { "kd", "cover", "r", "r-star",
        "ball", "x", "hilbert-r", "r-plus", "r-plus-plus", "vp", "rp", "max-rp",
        "ub", "oct" }, true, "unknown tree type");
    const string treeType = IO::GetParam<string>("tree_type");
    const bool randomBasis = IO::HasParam("random_basis");

    kfn = new KFNModel();

    KFNModel::TreeTypes tree = KFNModel::KD_TREE;
    if (treeType == "kd")
      tree = KFNModel::KD_TREE;
    else if (treeType == "cover")
      tree = KFNModel::COVER_TREE;
    else if (treeType == "r")
      tree = KFNModel::R_TREE;
    else if (treeType == "r-star")
      tree = KFNModel::R_STAR_TREE;
    else if (treeType == "ball")
      tree = KFNModel::BALL_TREE;
    else if (treeType == "x")
      tree = KFNModel::X_TREE;
    else if (treeType == "hilbert-r")
      tree = KFNModel::HILBERT_R_TREE;
    else if (treeType == "r-plus")
      tree = KFNModel::R_PLUS_TREE;
    else if (treeType == "r-plus-plus")
      tree = KFNModel::R_PLUS_PLUS_TREE;
    else if (treeType == "vp")
      tree = KFNModel::VP_TREE;
    else if (treeType == "rp")
      tree = KFNModel::RP_TREE;
    else if (treeType == "max-rp")
      tree = KFNModel::MAX_RP_TREE;
    else if (treeType == "ub")
      tree = KFNModel::UB_TREE;
    else if (treeType == "oct")
      tree = KFNModel::OCTREE;

    kfn->TreeType() = tree;
    kfn->RandomBasis() = randomBasis;

    Log::Info << "Using reference data from "
        << IO::GetPrintableParam<arma::mat>("reference") << "." << endl;

    arma::mat referenceSet = std::move(IO::GetParam<arma::mat>("reference"));

    kfn->BuildModel(std::move(referenceSet), size_t(lsInt), searchMode,
        epsilon);
  }
  else
  {
    // Load the model from file.
    kfn = IO::GetParam<KFNModel*>("input_model");

    // Adjust search mode.
    kfn->SearchMode() = searchMode;
    kfn->Epsilon() = epsilon;

    // If leaf_size wasn't provided, let's consider the current value in the
    // loaded model.  Else, update it (only considered when building the query
    // tree).
    if (IO::HasParam("leaf_size"))
      kfn->LeafSize() = size_t(lsInt);

    Log::Info << "Using kFN model from '"
        << IO::GetPrintableParam<KFNModel*>("input_model") << "' (trained on "
        << kfn->Dataset().n_rows << "x" << kfn->Dataset().n_cols
        << " dataset)." << endl;
  }

  // Perform search, if desired.
  if (IO::HasParam("k"))
  {
    const size_t k = (size_t) IO::GetParam<int>("k");

    arma::mat queryData;
    if (IO::HasParam("query"))
    {
      Log::Info << "Using query data from "
          << IO::GetPrintableParam<arma::mat>("query") << "." << endl;
      queryData = std::move(IO::GetParam<arma::mat>("query"));
      if (queryData.n_rows != kfn->Dataset().n_rows)
      {
        // Clean memory if needed.
        const size_t dimensions = kfn->Dataset().n_rows;
        if (IO::HasParam("reference"))
          delete kfn;
        Log::Fatal << "Query has invalid dimensions (" << queryData.n_rows <<
            "); should be " << dimensions << "!" << endl;
      }
    }

    // Sanity check on k value: must be greater than 0, must be less than or
    // equal to the number of reference points.  Since it is unsigned,
    // we only test the upper bound.
    if (k > kfn->Dataset().n_cols)
    {
      // Clean memory if needed.
      const size_t referencePoints = kfn->Dataset().n_cols;
      if (IO::HasParam("reference"))
        delete kfn;
      Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less "
          << "than or equal to the number of reference points ("
          << referencePoints << ")." << endl;
    }

    // Sanity check on k value: must not be equal to the number of reference
    // points when query data has not been provided.
    if (!IO::HasParam("query") && k == kfn->Dataset().n_cols)
    {
      // Clean memory if needed.
      const size_t referencePoints = kfn->Dataset().n_cols;
      if (IO::HasParam("reference"))
        delete kfn;
      Log::Fatal << "Invalid k: " << k << "; must be less than the number of "
          << "reference points (" << referencePoints << ") if query data has "
          << "not been provided." << endl;
    }

    // Now run the search.
    arma::Mat<size_t> neighbors;
    arma::mat distances;

    if (IO::HasParam("query"))
      kfn->Search(std::move(queryData), k, neighbors, distances);
    else
      kfn->Search(k, neighbors, distances);
    Log::Info << "Search complete." << endl;

    // Calculate the effective error, if desired.
    if (IO::HasParam("true_distances"))
    {
      if (kfn->Epsilon() == 0)
        Log::Warn << PRINT_PARAM_STRING("true_distances") << " specified, but "
            << "the search is exact, so there is no need to calculate the "
            << "error!" << endl;

      arma::mat trueDistances =
          std::move(IO::GetParam<arma::mat>("true_distances"));

      if (trueDistances.n_rows != distances.n_rows ||
          trueDistances.n_cols != distances.n_cols)
      {
        // Clean memory if needed.
        if (IO::HasParam("reference"))
          delete kfn;
        Log::Fatal << "The true distances file must have the same number of "
            << "values than the set of distances being queried!" << endl;
      }

      Log::Info << "Effective error: " << KFN::EffectiveError(distances,
          trueDistances) << endl;
    }

    // Calculate the recall, if desired.
    if (IO::HasParam("true_neighbors"))
    {
      if (kfn->Epsilon() == 0)
        Log::Warn << PRINT_PARAM_STRING("true_neighbors") << " specified, but "
            << "the search is exact, so there is no need to calculate the "
            << "recall!" << endl;

      arma::Mat<size_t> trueNeighbors =
          std::move(IO::GetParam<arma::Mat<size_t>>("true_neighbors"));

      if (trueNeighbors.n_rows != neighbors.n_rows ||
          trueNeighbors.n_cols != neighbors.n_cols)
      {
        // Clean memory if needed.
        if (IO::HasParam("reference"))
          delete kfn;
        Log::Fatal << "The true neighbors file must have the same number of "
            << "values than the set of neighbors being queried!" << endl;
      }

      Log::Info << "Recall: " << KFN::Recall(neighbors, trueNeighbors) << endl;
    }

    // Save output.
    IO::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    IO::GetParam<arma::mat>("distances") = std::move(distances);
  }

  IO::GetParam<KFNModel*>("output_model") = kfn;
}
