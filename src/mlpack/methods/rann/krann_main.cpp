/**
 * @file methods/rann/krann_main.cpp
 * @author Parikshit Ram
 *
 * Implementation of the kRANN executable.  Allows some number of standard
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

#include "ra_search.hpp"
#include "ra_model.hpp"
#include <mlpack/methods/neighbor_search/unmap.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

// Convenience typedef.
typedef RAModel<NearestNeighborSort> RANNModel;

// Program Name.
BINDING_NAME("K-Rank-Approximate-Nearest-Neighbors (kRANN)");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of rank-approximate k-nearest-neighbor search (kRANN) "
    " using single-tree and dual-tree algorithms.  Given a set of reference "
    "points and query points, this can find the k nearest neighbors in the "
    "reference set of each query point using trees; trees that are built can "
    "be saved for future use.");

// Long description.
BINDING_LONG_DESC(
    "This program will calculate the k rank-approximate-nearest-neighbors of a "
    "set of points. You may specify a separate set of reference points and "
    "query points, or just a reference set which will be used as both the "
    "reference and query set. You must specify the rank approximation (in %) "
    "(and optionally the success probability).");

// Example.
BINDING_EXAMPLE(
    "For example, the following will return 5 neighbors from the top 0.1% of "
    "the data (with probability 0.95) for each point in " +
    PRINT_DATASET("input") + " and store the distances in " +
    PRINT_DATASET("distances") + " and the neighbors in " +
    PRINT_DATASET("neighbors.csv") + ":"
    "\n\n" +
    PRINT_CALL("krann", "reference", "input", "k", 5, "distances", "distances",
        "neighbors", "neighbors", "tau", 0.1) +
    "\n\n"
    "Note that tau must be set such that the number of points in the "
    "corresponding percentile of the data is greater than k.  Thus, if we "
    "choose tau = 0.1 with a dataset of 1000 points and k = 5, then we are "
    "attempting to choose 5 nearest neighbors out of the closest 1 point -- "
    "this is invalid and the program will terminate with an error message."
    "\n\n"
    "The output matrices are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th nearest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// See also...
BINDING_SEE_ALSO("@knn", "#knn");
BINDING_SEE_ALSO("@lsh", "#lsh");
BINDING_SEE_ALSO("Rank-approximate nearest neighbor search: Retaining meaning"
        " and speed in high dimensions (pdf)", "https://papers.nips.cc/paper/"
        "3864-rank-approximate-nearest-neighbor-search-retaining-meaning-and"
        "-speed-in-high-dimensions.pdf");
BINDING_SEE_ALSO("mlpack::neighbor::RASearch C++ class documentation",
        "@doxygen/classmlpack_1_1neighbor_1_1RASearch.html");

// Define our input parameters that this program will take.
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");

// The option exists to load or save models.
PARAM_MODEL_IN(RANNModel, "input_model", "Pre-trained kNN model.", "m");
PARAM_MODEL_OUT(RANNModel, "output_model", "If specified, the kNN model will be"
    " output here.", "M");

// The user may specify a query file of query points and a number of nearest
// neighbors to search for.
PARAM_MATRIX_IN("query", "Matrix containing query points (optional).", "q");
PARAM_INT_IN("k", "Number of nearest neighbors to find.", "k", 0);

// The user may specify the type of tree to use, and a few parameters for tree
// building.
PARAM_STRING_IN("tree_type", "Type of tree to use: 'kd', 'ub', 'cover', 'r', "
    "'x', 'r-star', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'.", "t",
    "kd");
PARAM_INT_IN("leaf_size", "Leaf size for tree building (used for kd-trees, "
    "UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, "
    "R++ trees, and octrees).", "l", 20);
PARAM_FLAG("random_basis", "Before tree-building, project the data onto a "
    "random orthogonal basis.", "R");
PARAM_INT_IN("seed", "Random seed (if 0, std::time(NULL) is used).", "s", 0);

// Search options.
PARAM_DOUBLE_IN("tau", "The allowed rank-error in terms of the percentile of "
             "the data.", "T", 5);
PARAM_DOUBLE_IN("alpha", "The desired success probability.", "a", 0.95);
PARAM_FLAG("naive", "If true, sampling will be done without using a tree.",
           "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
           "dual-tree search.", "S");
PARAM_FLAG("sample_at_leaves", "The flag to trigger sampling at leaves.", "L");
PARAM_FLAG("first_leaf_exact", "The flag to trigger sampling only after "
           "exactly exploring the first leaf.", "X");
PARAM_INT_IN("single_sample_limit", "The limit on the maximum number of "
    "samples (and hence the largest node you can approximate).", "z", 20);

static void mlpackMain()
{
  if (IO::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) IO::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // A user cannot specify both reference data and a model.
  RequireOnlyOnePassed({ "reference", "input_model" }, true);

  ReportIgnoredParam({{ "input_model", true }}, "tree_type");
  ReportIgnoredParam({{ "input_model", true }}, "leaf_size");
  ReportIgnoredParam({{ "input_model", true }}, "random_basis");
  ReportIgnoredParam({{ "input_model", true }}, "naive");

  // The user should give something to do...
  RequireAtLeastOnePassed({ "k", "output_model" }, false, "no results will be "
      "saved");

  // If the user specifies k but no output files, they should be warned.
  if (IO::HasParam("k"))
  {
    RequireAtLeastOnePassed({ "neighbors", "distances" }, false, "no nearest "
        "neighbor search results will be saved");
  }

  // If the user specifies output files but no k, they should be warned.
  ReportIgnoredParam({{ "k", false }}, "neighbors");
  ReportIgnoredParam({{ "k", false }}, "distances");

  // Naive mode overrides single mode.
  ReportIgnoredParam({{ "naive", true }}, "single_mode");

  // Sanity check on leaf size.
  const int lsInt = IO::GetParam<int>("leaf_size");
  RequireParamValue<int>("leaf_size", [](int x) { return x > 0; }, true,
      "leaf size must be greater than 0");

  // Sanity check on tau.
  RequireParamValue<double>("tau", [](double x) {
      return (x >= 0.0 && x <=100.0); }, true,
      "tau must be in range [0.0, 100.0]");

  // Sanity check on alpha.
  RequireParamValue<double>("alpha", [](double x) {
      return (x >= 0.0 && x <=1.0); }, true,
      "alpha must be in range [0.0, 1.0]");

  // We either have to load the reference data, or we have to load the model.
  RANNModel* rann;
  const bool naive = IO::HasParam("naive");
  const bool singleMode = IO::HasParam("single_mode");
  if (IO::HasParam("reference"))
  {
    rann = new RANNModel();

    // Get all the parameters.
    const string treeType = IO::GetParam<string>("tree_type");
    RequireParamInSet<string>("tree_type", { "kd", "cover", "r", "r-star", "x",
        "hilbert-r", "r-plus", "r-plus-plus", "ub", "oct" }, true,
        "unknown tree type");
    const bool randomBasis = IO::HasParam("random_basis");

    RANNModel::TreeTypes tree = RANNModel::KD_TREE;
    if (treeType == "kd")
      tree = RANNModel::KD_TREE;
    else if (treeType == "cover")
      tree = RANNModel::COVER_TREE;
    else if (treeType == "r")
      tree = RANNModel::R_TREE;
    else if (treeType == "r-star")
      tree = RANNModel::R_STAR_TREE;
    else if (treeType == "x")
      tree = RANNModel::X_TREE;
    else if (treeType == "hilbert-r")
      tree = RANNModel::HILBERT_R_TREE;
    else if (treeType == "r-plus")
      tree = RANNModel::R_PLUS_TREE;
    else if (treeType == "r-plus-plus")
      tree = RANNModel::R_PLUS_PLUS_TREE;
    else if (treeType == "ub")
      tree = RANNModel::UB_TREE;
    else if (treeType == "oct")
      tree = RANNModel::OCTREE;

    rann->TreeType() = tree;
    rann->RandomBasis() = randomBasis;

    Log::Info << "Using reference data from "
        << IO::GetPrintableParam<arma::mat>("reference") << "." << endl;
    arma::mat referenceSet = std::move(IO::GetParam<arma::mat>("reference"));

    rann->BuildModel(std::move(referenceSet), size_t(lsInt), naive, singleMode);
  }
  else
  {
    // Load the model from file.
    rann = IO::GetParam<RANNModel*>("input_model");

    Log::Info << "Using rank-approximate kNN model from '"
        << IO::GetPrintableParam<RANNModel*>("input_model") << "' (trained on "
        << rann->Dataset().n_rows << "x" << rann->Dataset().n_cols
        << " dataset)." << endl;

    // Adjust singleMode and naive if necessary.
    rann->SingleMode() = IO::HasParam("single_mode");
    rann->Naive() = IO::HasParam("naive");
    rann->LeafSize() = size_t(lsInt);
  }

  // Apply the parameters for search.
  if (IO::HasParam("tau"))
    rann->Tau() = IO::GetParam<double>("tau");
  if (IO::HasParam("alpha"))
    rann->Alpha() = IO::GetParam<double>("alpha");
  if (IO::HasParam("single_sample_limit"))
    rann->SingleSampleLimit() = IO::GetParam<int>("single_sample_limit");
  rann->SampleAtLeaves() = IO::HasParam("sample_at_leaves");
  rann->FirstLeafExact() = IO::HasParam("sample_at_leaves");

  // Perform search, if desired.
  if (IO::HasParam("k"))
  {
    const size_t k = (size_t) IO::GetParam<int>("k");

    arma::mat queryData;
    if (IO::HasParam("query"))
    {
      queryData = std::move(IO::GetParam<arma::mat>("query"));
      Log::Info << "Using query data from '"
          << IO::GetPrintableParam<arma::mat>("query") << "' ("
          << queryData.n_rows << "x" << queryData.n_cols << ")." << endl;
      if (queryData.n_rows != rann->Dataset().n_rows)
      {
        const size_t dimensions = rann->Dataset().n_rows;
        Log::Fatal << "Query has invalid dimensions(" << queryData.n_rows <<
            "); should be " << dimensions << "!" << endl;
      }
    }

    // Sanity check on k value: must be greater than 0, must be less than the
    // number of reference points.  Since it is unsigned, we only test the upper
    // bound.
    if (k > rann->Dataset().n_cols)
    {
      Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less ";
      Log::Fatal << "than or equal to the number of reference points (";
      Log::Fatal << rann->Dataset().n_cols << ")." << endl;
    }

    arma::Mat<size_t> neighbors;
    arma::mat distances;
    if (IO::HasParam("query"))
      rann->Search(std::move(queryData), k, neighbors, distances);
    else
      rann->Search(k, neighbors, distances);
    Log::Info << "Search complete." << endl;

    // Save output.
    IO::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    IO::GetParam<arma::mat>("distances") = std::move(distances);
  }

  // Save the output model.
  IO::GetParam<RANNModel*>("output_model") = rann;
}
