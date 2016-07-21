/**
w
 * @file kfn_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of the KFN executable.  Allows some number of standard
 * options.
 *
 * This file is part of mlpack 2.0.3.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

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
    "$ mlpack_kfn --k=5 --reference_file=input.csv "
    "--distances_file=distances.csv\n --neighbors_file=neighbors.csv"
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th furthest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// Define our input parameters that this program will take.
PARAM_STRING("reference_file", "File containing the reference dataset.", "r",
    "");
PARAM_STRING("distances_file", "File to output distances into.", "d", "");
PARAM_STRING("neighbors_file", "File to output neighbors into.", "n", "");

// The option exists to load or save models.
PARAM_STRING("input_model_file", "File containing pre-trained kFN model.", "m",
    "");
PARAM_STRING("output_model_file", "If specified, the kFN model will be saved to"
    " the given file.", "M", "");

// The user may specify a query file of query points and a number of furthest
// neighbors to search for.
PARAM_STRING("query_file", "File containing query points (optional).", "q", "");
PARAM_INT("k", "Number of furthest neighbors to find.", "k", 0);

// The user may specify the type of tree to use, and a few pararmeters for tree
// building.
PARAM_STRING("tree_type", "Type of tree to use: 'kd', 'cover', 'r', 'r-star', "
    "'x', 'ball'.", "t", "kd");
PARAM_INT("leaf_size", "Leaf size for tree building.", "l", 20);
PARAM_FLAG("random_basis", "Before tree-building, project the data onto a "
    "random orthogonal basis.", "R");
PARAM_INT("seed", "Random seed (if 0, std::time(NULL) is used).", "s", 0);

// Search settings.
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
    "dual-tree search).", "s");

// Convenience typedef.
typedef NSModel<FurthestNeighborSort> KFNModel;

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
        << "so the furthest neighbor search results will not be saved!" << endl;

  // If the user specifies output files but no k, they should be warned.
  if ((CLI::HasParam("neighbors_file") || CLI::HasParam("distances_file")) &&
      !CLI::HasParam("k"))
    Log::Warn << "An output file for furthest neighbor search is given ("
        << "--neighbors_file or --distances_file), but furthest neighbor search"
        << " is not being performed because k (--k) is not specified!  No "
        << "results will be saved." << endl;

  // Sanity check on leaf size.
  const int lsInt = CLI::GetParam<int>("leaf_size");
  if (lsInt < 1)
    Log::Fatal << "Invalid leaf size: " << lsInt << ".  Must be greater than 0."
        << endl;

  // We either have to load the reference data, or we have to load the model.
  NSModel<FurthestNeighborSort> kfn;
  const bool naive = CLI::HasParam("naive");
  const bool singleMode = CLI::HasParam("single_mode");
  if (CLI::HasParam("reference_file"))
  {
    // Get all the parameters.
    const string referenceFile = CLI::GetParam<string>("reference_file");
    const string treeType = CLI::GetParam<string>("tree_type");
    const bool randomBasis = CLI::HasParam("random_basis");

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
    else
      Log::Fatal << "Unknown tree type '" << treeType << "'; valid choices are "
          << "'kd', 'cover', 'r', 'r-star', 'x' and 'ball'." << endl;

    kfn.TreeType() = tree;
    kfn.RandomBasis() = randomBasis;

    arma::mat referenceSet;
    data::Load(referenceFile, referenceSet, true);

    Log::Info << "Loaded reference data from '" << referenceFile << "' ("
        << referenceSet.n_rows << "x" << referenceSet.n_cols << ")." << endl;

    kfn.BuildModel(std::move(referenceSet), size_t(lsInt), naive, singleMode);
  }
  else
  {
    // Load the model from file.
    const string inputModelFile = CLI::GetParam<string>("input_model_file");
    data::Load(inputModelFile, "kfn_model", kfn, true); // Fatal on failure.

    Log::Info << "Loaded kFN model from '" << inputModelFile << "' (trained on "
        << kfn.Dataset().n_rows << "x" << kfn.Dataset().n_cols << " dataset)."
        << endl;

    // Adjust singleMode and naive if necessary.
    kfn.SingleMode() = CLI::HasParam("single_mode");
    kfn.Naive() = CLI::HasParam("naive");
    kfn.LeafSize() = size_t(lsInt);
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
    if (k > kfn.Dataset().n_cols)
    {
      Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less "
          << "than or equal to the number of reference points ("
          << kfn.Dataset().n_cols << ")." << endl;
    }

    // Naive mode overrides single mode.
    if (singleMode && naive)
      Log::Warn << "--single_mode ignored because --naive is present." << endl;

    // Now run the search.
    arma::Mat<size_t> neighbors;
    arma::mat distances;

    if (CLI::HasParam("query_file"))
      kfn.Search(std::move(queryData), k, neighbors, distances);
    else
      kfn.Search(k, neighbors, distances);
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
    data::Save(outputModelFile, "kfn_model", kfn);
  }
}
