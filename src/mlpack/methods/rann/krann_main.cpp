/**
 * @file allkrann_main.cpp
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

#include "ra_search.hpp"
#include "ra_model.hpp"
#include <mlpack/methods/neighbor_search/unmap.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;

// Information about the program itself.
PROGRAM_INFO("K-Rank-Approximate-Nearest-Neighbors (kRANN)",
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
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");

// The option exists to load or save models.
PARAM_STRING_IN("input_model_file", "File containing pre-trained kNN model.",
    "m", "");
PARAM_STRING_OUT("output_model_file", "If specified, the kNN model will be "
    "saved to the given file.", "M");

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

// Convenience typedef.
typedef RAModel<NearestNeighborSort> RANNModel;

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

  // The user should give something to do...
  if (!CLI::HasParam("k") && !CLI::HasParam("output_model_file"))
    Log::Warn << "Neither -k nor --output_model_file are specified, so no "
        << "results from this program will be saved!" << endl;

  // If the user specifies k but no output files, they should be warned.
  if (CLI::HasParam("k") &&
      !(CLI::HasParam("neighbors") || CLI::HasParam("distances")))
    Log::Warn << "Neither --neighbors_file nor --distances_file is specified, "
        << "so the nearest neighbor search results will not be saved!" << endl;

  // If the user specifies output files but no k, they should be warned.
  if ((CLI::HasParam("neighbors") || CLI::HasParam("distances")) &&
      !CLI::HasParam("k"))
    Log::Warn << "An output file for nearest neighbor search is given ("
        << "--neighbors_file or --distances_file), but nearest neighbor search "
        << "is not being performed because k (--k) is not specified!  No "
        << "results will be saved." << endl;

  // Sanity check on leaf size.
  const int lsInt = CLI::GetParam<int>("leaf_size");
  if (lsInt < 1)
  {
    Log::Fatal << "Invalid leaf size: " << lsInt << ".  Must be greater "
        "than 0." << endl;
  }

  // We either have to load the reference data, or we have to load the model.
  RANNModel rann;
  const bool naive = CLI::HasParam("naive");
  const bool singleMode = CLI::HasParam("single_mode");
  if (CLI::HasParam("reference"))
  {
    // Get all the parameters.
    const string treeType = CLI::GetParam<string>("tree_type");
    const bool randomBasis = CLI::HasParam("random_basis");

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
    else
      Log::Fatal << "Unknown tree type '" << treeType << "'; valid choices are "
          << "'kd', 'ub', 'cover', 'r', 'r-star', 'x', 'hilbert-r', "
          << "'r-plus', 'r-plus-plus', 'oct'." << endl;

    rann.TreeType() = tree;
    rann.RandomBasis() = randomBasis;

    arma::mat referenceSet = std::move(CLI::GetParam<arma::mat>("reference"));

    Log::Info << "Loaded reference data from '"
        << CLI::GetUnmappedParam<arma::mat>("reference") << "' ("
        << referenceSet.n_rows << " x " << referenceSet.n_cols << ")."
        << endl;

    rann.BuildModel(std::move(referenceSet), size_t(lsInt), naive, singleMode);
  }
  else
  {
    // Load the model from file.
    const string inputModelFile = CLI::GetParam<string>("input_model_file");
    data::Load(inputModelFile, "rann_model", rann, true); // Fatal on failure.

    Log::Info << "Loaded rank-approximate kNN model from '" << inputModelFile
        << "' (trained on " << rann.Dataset().n_rows << "x"
        << rann.Dataset().n_cols << " dataset)." << endl;

    // Adjust singleMode and naive if necessary.
    rann.SingleMode() = CLI::HasParam("single_mode");
    rann.Naive() = CLI::HasParam("naive");
    rann.LeafSize() = size_t(lsInt);
  }

  // Apply the parameters for search.
  if (CLI::HasParam("tau"))
    rann.Tau() = CLI::GetParam<double>("tau");
  if (CLI::HasParam("alpha"))
    rann.Alpha() = CLI::GetParam<double>("alpha");
  if (CLI::HasParam("single_sample_limit"))
    rann.SingleSampleLimit() = CLI::GetParam<double>("single_sample_limit");
  rann.SampleAtLeaves() = CLI::HasParam("sample_at_leaves");
  rann.FirstLeafExact() = CLI::HasParam("sample_at_leaves");

  // Perform search, if desired.
  if (CLI::HasParam("k"))
  {
    const size_t k = (size_t) CLI::GetParam<int>("k");

    arma::mat queryData;
    if (CLI::HasParam("query"))
    {
      queryData = std::move(CLI::GetParam<arma::mat>("query"));
      Log::Info << "Loaded query data from '"
          << CLI::GetUnmappedParam<arma::mat>("query") << "' ("
          << queryData.n_rows << "x" << queryData.n_cols << ")." << endl;
    }

    // Sanity check on k value: must be greater than 0, must be less than the
    // number of reference points.  Since it is unsigned, we only test the upper
    // bound.
    if (k > rann.Dataset().n_cols)
    {
      Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less ";
      Log::Fatal << "than or equal to the number of reference points (";
      Log::Fatal << rann.Dataset().n_cols << ")." << endl;
    }

    // Naive mode overrides single mode.
    if (singleMode && naive)
    {
      Log::Warn << "--single_mode ignored because --naive is present." << endl;
    }

    arma::Mat<size_t> neighbors;
    arma::mat distances;
    if (CLI::HasParam("query"))
      rann.Search(std::move(queryData), k, neighbors, distances);
    else
      rann.Search(k, neighbors, distances);
    Log::Info << "Search complete." << endl;

    // Save output, if desired.
    if (CLI::HasParam("neighbors"))
      CLI::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    if (CLI::HasParam("distances"))
      CLI::GetParam<arma::mat>("distances") = std::move(distances);
  }

  if (CLI::HasParam("output_model_file"))
  {
    const string outputModelFile = CLI::GetParam<string>("output_model_file");
    data::Save(outputModelFile, "rann_model", rann);
  }
}
