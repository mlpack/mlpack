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
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME krann

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/neighbor_search/unmap.hpp>
#include "ra_search.hpp"
#include "ra_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("K-Rank-Approximate-Nearest-Neighbors (kRANN)");

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
    " and speed in high dimensions (pdf)", "https://proceedings.neurips.cc/"
    "paper_files/paper/2009/file/ddb30680a691d157187ee1cf9e896d03-Paper.pdf");
BINDING_SEE_ALSO("RASearch C++ class documentation",
    "@src/mlpack/methods/rann/ra_search.hpp");

// Define our input parameters that this program will take.
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");

// The option exists to load or save models.
PARAM_MODEL_IN(RAModel, "input_model", "Pre-trained kNN model.", "m");
PARAM_MODEL_OUT(RAModel, "output_model", "If specified, the kNN model will be"
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

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  // A user cannot specify both reference data and a model.
  RequireOnlyOnePassed(params, { "reference", "input_model" }, true);

  ReportIgnoredParam(params, {{ "input_model", true }}, "tree_type");
  ReportIgnoredParam(params, {{ "input_model", true }}, "leaf_size");
  ReportIgnoredParam(params, {{ "input_model", true }}, "random_basis");
  ReportIgnoredParam(params, {{ "input_model", true }}, "naive");

  // The user should give something to do...
  RequireAtLeastOnePassed(params, { "k", "output_model" }, false,
      "no results will be saved");

  // If the user specifies k but no output files, they should be warned.
  if (params.Has("k"))
  {
    RequireAtLeastOnePassed(params, { "neighbors", "distances" }, false,
        "no nearest neighbor search results will be saved");
  }

  // If the user specifies output files but no k, they should be warned.
  ReportIgnoredParam(params, {{ "k", false }}, "neighbors");
  ReportIgnoredParam(params, {{ "k", false }}, "distances");

  // Naive mode overrides single mode.
  ReportIgnoredParam(params, {{ "naive", true }}, "single_mode");

  // Sanity check on leaf size.
  const int lsInt = params.Get<int>("leaf_size");
  RequireParamValue<int>(params, "leaf_size", [](int x) { return x > 0; }, true,
      "leaf size must be greater than 0");

  // Sanity check on tau.
  RequireParamValue<double>(params, "tau", [](double x) {
      return (x >= 0.0 && x <=100.0); }, true,
      "tau must be in range [0.0, 100.0]");

  // Sanity check on alpha.
  RequireParamValue<double>(params, "alpha", [](double x) {
      return (x >= 0.0 && x <=1.0); }, true,
      "alpha must be in range [0.0, 1.0]");

  // We either have to load the reference data, or we have to load the model.
  RAModel* rann;
  const bool naive = params.Has("naive");
  const bool singleMode = params.Has("single_mode");
  if (params.Has("reference"))
  {
    rann = new RAModel();

    // Get all the parameters.
    const string treeType = params.Get<string>("tree_type");
    RequireParamInSet<string>(params, "tree_type", { "kd", "cover", "r",
        "r-star", "x", "hilbert-r", "r-plus", "r-plus-plus", "ub", "oct" },
        true, "unknown tree type");
    const bool randomBasis = params.Has("random_basis");

    RAModel::TreeTypes tree = RAModel::KD_TREE;
    if (treeType == "kd")
      tree = RAModel::KD_TREE;
    else if (treeType == "cover")
      tree = RAModel::COVER_TREE;
    else if (treeType == "r")
      tree = RAModel::R_TREE;
    else if (treeType == "r-star")
      tree = RAModel::R_STAR_TREE;
    else if (treeType == "x")
      tree = RAModel::X_TREE;
    else if (treeType == "hilbert-r")
      tree = RAModel::HILBERT_R_TREE;
    else if (treeType == "r-plus")
      tree = RAModel::R_PLUS_TREE;
    else if (treeType == "r-plus-plus")
      tree = RAModel::R_PLUS_PLUS_TREE;
    else if (treeType == "ub")
      tree = RAModel::UB_TREE;
    else if (treeType == "oct")
      tree = RAModel::OCTREE;

    rann->TreeType() = tree;
    rann->RandomBasis() = randomBasis;

    arma::mat& referenceSet = params.Get<arma::mat>("reference");

    Log::Info << "Using reference data from "
        << params.GetPrintable<arma::mat>("reference") << "." << endl;

    rann->BuildModel(timers, std::move(referenceSet), size_t(lsInt), naive,
        singleMode);
  }
  else
  {
    // Load the model from file.
    rann = params.Get<RAModel*>("input_model");

    Log::Info << "Using rank-approximate kNN model from '"
        << params.GetPrintable<RAModel*>("input_model") << "' (trained on "
        << rann->Dataset().n_rows << "x" << rann->Dataset().n_cols
        << " dataset)." << endl;

    // Adjust singleMode and naive if necessary.
    rann->SingleMode() = params.Has("single_mode");
    rann->Naive() = params.Has("naive");
    rann->LeafSize() = size_t(lsInt);
  }

  // Apply the parameters for search.
  if (params.Has("tau"))
    rann->Tau() = params.Get<double>("tau");
  if (params.Has("alpha"))
    rann->Alpha() = params.Get<double>("alpha");
  if (params.Has("single_sample_limit"))
    rann->SingleSampleLimit() = params.Get<int>("single_sample_limit");
  rann->SampleAtLeaves() = params.Has("sample_at_leaves");
  rann->FirstLeafExact() = params.Has("sample_at_leaves");

  // Perform search, if desired.
  if (params.Has("k"))
  {
    const size_t k = (size_t) params.Get<int>("k");

    arma::mat queryData;
    if (params.Has("query"))
    {
      // Workaround: this avoids printing load information twice for the CLI
      // bindings, where GetPrintable() will trigger a call to data::Load(),
      // which prints loading information in the middle of the Log::Info
      // message.
      (void) params.Get<arma::mat>("query");

      queryData = std::move(params.Get<arma::mat>("query"));
      Log::Info << "Using query data from '"
          << params.GetPrintable<arma::mat>("query") << "' ("
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
    if (params.Has("query"))
      rann->Search(timers, std::move(queryData), k, neighbors, distances);
    else
      rann->Search(timers, k, neighbors, distances);
    Log::Info << "Search complete." << endl;

    // Save output.
    params.Get<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    params.Get<arma::mat>("distances") = std::move(distances);
  }

  // Save the output model.
  params.Get<RAModel*>("output_model") = rann;
}
