/**
 * @file knn_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of the AllkNN executable.  Allows some number of standard
 * options.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

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

// Information about the program itself.
PROGRAM_INFO("k-Nearest-Neighbors",
    "This program will calculate the k-nearest-neighbors of a set of "
    "points using kd-trees or cover trees (cover tree support is experimental "
    "and may be slow). You may specify a separate set of "
    "reference points and query points, or just a reference set which will be "
    "used as both the reference and query set."
    "\n\n"
    "For example, the following will calculate the 5 nearest neighbors of each"
    "point in 'input.csv' and store the distances in 'distances.csv' and the "
    "neighbors in the file 'neighbors.csv':"
    "\n\n"
    "$ mlpack_knn --k=5 --reference_file=input.csv "
    "--distances_file=distances.csv\n --neighbors_file=neighbors.csv"
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th nearest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// Define our input parameters that this program will take.
PARAM_STRING_IN("reference_file", "File containing the reference dataset.", "r",
    "");
PARAM_STRING_OUT("distances_file", "File to output distances into.", "d");
PARAM_STRING_OUT("neighbors_file", "File to output neighbors into.", "n");

// The option exists to load or save models.
PARAM_STRING_IN("input_model_file", "File containing pre-trained kNN model.",
    "m", "");
PARAM_STRING_OUT("output_model_file", "If specified, the kNN model will be "
    "saved to the given file.", "M");

// The user may specify a query file of query points and a number of nearest
// neighbors to search for.
PARAM_STRING_IN("query_file", "File containing query points (optional).", "q",
    "");
PARAM_INT_IN("k", "Number of nearest neighbors to find.", "k", 0);

// The user may specify the type of tree to use, and a few parameters for tree
// building.
PARAM_STRING_IN("tree_type", "Type of tree to use: 'kd', 'cover', 'r', "
    "'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'spill'.",
    "t", "kd");
PARAM_INT_IN("leaf_size", "Leaf size for tree building (used for kd-trees, R "
    "trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and Spill "
    "trees).", "l", 20);
PARAM_DOUBLE_IN("tau", "Overlapping size (for spill trees).", "u", 0);
PARAM_DOUBLE_IN("rho", "Balance threshold (for spill trees).", "b", 0.7);

PARAM_FLAG("random_basis", "Before tree-building, project the data onto a "
    "random orthogonal basis.", "R");
PARAM_INT_IN("seed", "Random seed (if 0, std::time(NULL) is used).", "s", 0);

// Search settings.
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
    "dual-tree search).", "S");
PARAM_DOUBLE_IN("epsilon", "If specified, will do approximate nearest neighbor "
    "search with given relative error.", "e", 0);

// Convenience typedef.
typedef NSModel<NearestNeighborSort> KNNModel;

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // A user cannot specify both reference data and a model.
  if (CLI::HasParam("reference_file") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Only one of --reference_file (-r) or --input_model_file (-m)"
        << " may be specified!" << endl;

  // A user must specify one of them...
  if (!CLI::HasParam("reference_file") && !CLI::HasParam("input_model_file"))
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
    if (CLI::HasParam("tau"))
      Log::Warn << "--tau (-u) will be ignored because --input_model_file"
          << " is specified." << endl;
    if (CLI::HasParam("rho"))
      Log::Warn << "--rho (-b) will be ignored because --input_model_file"
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
      !(CLI::HasParam("neighbors_file") || CLI::HasParam("distances_file")))
    Log::Warn << "Neither --neighbors_file nor --distances_file is specified, "
        << "so the nearest neighbor search results will not be saved!" << endl;

  // If the user specifies output files but no k, they should be warned.
  if ((CLI::HasParam("neighbors_file") || CLI::HasParam("distances_file")) &&
      !CLI::HasParam("k"))
    Log::Warn << "An output file for nearest neighbor search is given ("
        << "--neighbors_file or --distances_file), but nearest neighbor search "
        << "is not being performed because k (--k) is not specified!  No "
        << "results will be saved." << endl;

  // Sanity check on leaf size.
  const int lsInt = CLI::GetParam<int>("leaf_size");
  if (lsInt < 1)
    Log::Fatal << "Invalid leaf size: " << lsInt << ".  Must be greater "
        "than 0." << endl;

  // Sanity check on tau.
  const double tau = CLI::GetParam<double>("tau");
  if (tau < 0)
    Log::Fatal << "Invalid tau: " << tau << ".  Must be non-negative. " << endl;
  if (CLI::HasParam("tau") && "spill" != CLI::GetParam<string>("tree_type"))
    Log::Fatal << "Tau parameter is only valid for spill trees." << endl;

  // Sanity check on rho.
  const double rho = CLI::GetParam<double>("rho");
  if (rho < 0 || rho > 1)
    Log::Fatal << "Invalid rho: " << rho << ".  Must be in the range [0,1]. "
        << endl;
  if (CLI::HasParam("rho") && "spill" != CLI::GetParam<string>("tree_type"))
    Log::Fatal << "Rho parameter is only valid for spill trees." << endl;

  // Sanity check on epsilon.
  const double epsilon = CLI::GetParam<double>("epsilon");
  if (epsilon < 0)
    Log::Fatal << "Invalid epsilon: " << epsilon << ".  Must be non-negative. "
        << endl;

  // We either have to load the reference data, or we have to load the model.
  NSModel<NearestNeighborSort> knn;
  const bool naive = CLI::HasParam("naive");
  const bool singleMode = CLI::HasParam("single_mode");
  if (CLI::HasParam("reference_file"))
  {
    // Get all the parameters.
    const string referenceFile = CLI::GetParam<string>("reference_file");
    const string treeType = CLI::GetParam<string>("tree_type");
    const bool randomBasis = CLI::HasParam("random_basis");

    KNNModel::TreeTypes tree = KNNModel::KD_TREE;
    if (treeType == "kd")
      tree = KNNModel::KD_TREE;
    else if (treeType == "cover")
      tree = KNNModel::COVER_TREE;
    else if (treeType == "r")
      tree = KNNModel::R_TREE;
    else if (treeType == "r-star")
      tree = KNNModel::R_STAR_TREE;
    else if (treeType == "ball")
      tree = KNNModel::BALL_TREE;
    else if (treeType == "x")
      tree = KNNModel::X_TREE;
    else if (treeType == "hilbert-r")
      tree = KNNModel::HILBERT_R_TREE;
    else if (treeType == "r-plus")
      tree = KNNModel::R_PLUS_TREE;
    else if (treeType == "r-plus-plus")
      tree = KNNModel::R_PLUS_PLUS_TREE;
    else if (treeType == "spill")
      tree = KNNModel::SPILL_TREE;
    else
      Log::Fatal << "Unknown tree type '" << treeType << "'; valid choices are "
          << "'kd', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', "
          << "'r-plus', 'r-plus-plus' and 'spill'." << endl;

    knn.TreeType() = tree;
    knn.RandomBasis() = randomBasis;
    knn.LeafSize() = size_t(lsInt);
    knn.Tau() = tau;
    knn.Rho() = rho;

    arma::mat referenceSet;
    data::Load(referenceFile, referenceSet, true);

    Log::Info << "Loaded reference data from '" << referenceFile << "' ("
        << referenceSet.n_rows << " x " << referenceSet.n_cols << ")."
        << endl;

    knn.BuildModel(std::move(referenceSet), size_t(lsInt), naive, singleMode,
        epsilon);
  }
  else
  {
    // Load the model from file.
    const string inputModelFile = CLI::GetParam<string>("input_model_file");
    data::Load(inputModelFile, "knn_model", knn, true); // Fatal on failure.

    Log::Info << "Loaded kNN model from '" << inputModelFile << "' (trained on "
        << knn.Dataset().n_rows << "x" << knn.Dataset().n_cols << " dataset)."
        << endl;

    // Adjust singleMode and naive if necessary.
    knn.SingleMode() = CLI::HasParam("single_mode");
    knn.Naive() = CLI::HasParam("naive");
    knn.LeafSize() = size_t(lsInt);
    knn.Epsilon() = epsilon;
    knn.Tau() = tau;
    knn.Rho() = rho;
  }

  // Perform search, if desired.
  if (CLI::HasParam("k"))
  {
    const string queryFile = CLI::GetParam<string>("query_file");
    const size_t k = (size_t) CLI::GetParam<int>("k");

    arma::mat queryData;
    if (queryFile != "")
    {
      data::Load(queryFile, queryData, true);
      Log::Info << "Loaded query data from '" << queryFile << "' ("
          << queryData.n_rows << "x" << queryData.n_cols << ")." << endl;
    }

    // Sanity check on k value: must be greater than 0, must be less than the
    // number of reference points.  Since it is unsigned, we only test the upper
    // bound.
    if (k > knn.Dataset().n_cols)
    {
      Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less ";
      Log::Fatal << "than or equal to the number of reference points (";
      Log::Fatal << knn.Dataset().n_cols << ")." << endl;
    }

    // Naive mode overrides single mode.
    if (singleMode && naive)
    {
      Log::Warn << "--single_mode ignored because --naive is present." << endl;
    }

    // Now run the search.
    arma::Mat<size_t> neighbors;
    arma::mat distances;

    if (CLI::HasParam("query_file"))
      knn.Search(std::move(queryData), k, neighbors, distances);
    else
      knn.Search(k, neighbors, distances);
    Log::Info << "Search complete." << endl;

    // Save output, if desired.
    if (CLI::HasParam("neighbors_file"))
      data::Save(CLI::GetParam<string>("neighbors_file"), neighbors);
    if (CLI::HasParam("distances_file"))
      data::Save(CLI::GetParam<string>("distances_file"), distances);
  }

  if (CLI::HasParam("output_model_file"))
  {
    const string outputModelFile = CLI::GetParam<string>("output_model_file");
    data::Save(outputModelFile, "knn_model", knn);
  }
}
