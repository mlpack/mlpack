/**
 * @file lsh_main.cpp
 * @author Parikshit Ram
 *
 * This file computes the approximate nearest-neighbors using 2-stable
 * Locality-sensitive Hashing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include "lsh_search.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;

// Information about the program itself.
PROGRAM_INFO("All K-Approximate-Nearest-Neighbor Search with LSH",
    "This program will calculate the k approximate-nearest-neighbors of a set "
    "of points using locality-sensitive hashing. You may specify a separate set"
    " of reference points and query points, or just a reference set which will "
    "be used as both the reference and query set. "
    "\n\n"
    "For example, the following will return 5 neighbors from the data for each "
    "point in 'input.csv' and store the distances in 'distances.csv' and the "
    "neighbors in the file 'neighbors.csv':"
    "\n\n"
    "$ lsh -k 5 -r input.csv -d distances.csv -n neighbors.csv "
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th nearest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points."
    "\n\n"
    "Because this is approximate-nearest-neighbors search, results may be "
    "different from run to run.  Thus, the --seed option can be specified to "
    "set the random seed.");

// Define our input parameters that this program will take.
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");

// We can load or save models.
PARAM_STRING_IN("input_model_file", "File to load LSH model from.  (Cannot be "
    "specified with --reference_file.)", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save LSH model to.", "M");

// For testing recall.
PARAM_UMATRIX_IN("true_neighbors", "Matrix of true neighbors to compute "
    "recall with (the recall is printed when -v is specified).", "t");

PARAM_INT_IN("k", "Number of nearest neighbors to find.", "k", 0);
PARAM_MATRIX_IN("query", "Matrix containing query points (optional).", "q");

PARAM_INT_IN("projections", "The number of hash functions for each table", "K",
    10);
PARAM_INT_IN("tables", "The number of hash tables to be used.", "L", 30);
PARAM_DOUBLE_IN("hash_width", "The hash width for the first-level hashing in "
    "the LSH preprocessing. By default, the LSH class automatically estimates "
    "a hash width for its use.", "H", 0.0);
PARAM_INT_IN("num_probes", "Number of additional probes for multiprobe LSH; if "
    "0, traditional LSH is used.", "T", 0);
PARAM_INT_IN("second_hash_size", "The size of the second level hash table.",
    "S", 99901);
PARAM_INT_IN("bucket_size", "The size of a bucket in the second level hash.",
    "B", 500);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) time(NULL));

  // Get all the parameters.
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");

  size_t k = CLI::GetParam<int>("k");
  size_t secondHashSize = CLI::GetParam<int>("second_hash_size");
  size_t bucketSize = CLI::GetParam<int>("bucket_size");

  if (CLI::HasParam("input_model_file") && CLI::HasParam("reference"))
  {
    Log::Fatal << "Cannot specify both --reference_file and --input_model_file!"
        << " Either create a new model with --reference_file or use an existing"
        << " model with --input_model_file." << endl;
  }

  if (!CLI::HasParam("input_model_file") && !CLI::HasParam("reference"))
  {
    Log::Fatal << "Must specify either --input_model_file or --reference_file!"
        << endl;
  }

  if (!CLI::HasParam("neighbors") && !CLI::HasParam("distances") &&
      !CLI::HasParam("output_model_file"))
  {
    Log::Warn << "Neither --neighbors_file, --distances_file, nor "
        << "--output_model_file are specified; no results will be saved."
        << endl;
  }

  if ((CLI::HasParam("query") && !CLI::HasParam("k")) ||
      (!CLI::HasParam("query") && !CLI::HasParam("reference") &&
       CLI::HasParam("k")))
  {
    Log::Fatal << "Both --query_file or --reference_file and --k must be "
        << "specified if search is to be done!" << endl;
  }

  if (CLI::HasParam("input_model_file") && CLI::HasParam("k") &&
      !CLI::HasParam("query"))
  {
    Log::Info << "Performing LSH-based approximate nearest neighbor search on "
        << "the reference dataset in the model stored in '" << inputModelFile
        << "'." << endl;
  }

  if (!CLI::HasParam("k") && CLI::HasParam("neighbors"))
    Log::Warn << "--neighbors_file ignored because --k is not specified."
        << endl;

  if (!CLI::HasParam("k") && CLI::HasParam("distances"))
    Log::Warn << "--distances_file ignored because --k is not specified."
        << endl;

  // These declarations are here so that the matrices don't go out of scope.
  arma::mat referenceData;
  arma::mat queryData;

  // Pick up the LSH-specific parameters.
  const size_t numProj = CLI::GetParam<int>("projections");
  const size_t numTables = CLI::GetParam<int>("tables");
  const double hashWidth = CLI::GetParam<double>("hash_width");
  const size_t numProbes = (size_t) CLI::GetParam<int>("num_probes");

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (hashWidth == 0.0)
    Log::Info << "Using LSH with " << numProj << " projections (K) and " <<
        numTables << " tables (L) with default hash width." << endl;
  else
    Log::Info << "Using LSH with " << numProj << " projections (K) and " <<
        numTables << " tables (L) with hash width(r): " << hashWidth << endl;

  LSHSearch<> allkann;
  if (CLI::HasParam("reference"))
  {
    referenceData = std::move(CLI::GetParam<arma::mat>("reference"));
    Log::Info << "Loaded reference data from '"
        << CLI::GetUnmappedParam<arma::mat>("reference") << "' ("
        << referenceData.n_rows << " x " << referenceData.n_cols << ")."
        << endl;

    Timer::Start("hash_building");
    allkann.Train(referenceData, numProj, numTables, hashWidth, secondHashSize,
        bucketSize);
    Timer::Stop("hash_building");
  }
  else if (CLI::HasParam("input_model_file"))
  {
    data::Load(inputModelFile, "lsh_model", allkann, true); // Fatal on fail.
  }

  if (CLI::HasParam("k"))
  {
    Log::Info << "Computing " << k << " distance approximate nearest neighbors."
        << endl;
    if (CLI::HasParam("query"))
    {
      queryData = std::move(CLI::GetParam<arma::mat>("query"));
      Log::Info << "Loaded query data from '"
          << CLI::GetUnmappedParam<arma::mat>("query") << "' ("
          << queryData.n_rows << " x " << queryData.n_cols << ")." << endl;

      allkann.Search(queryData, k, neighbors, distances, 0, numProbes);
    }
    else
    {
      allkann.Search(k, neighbors, distances, 0, numProbes);
    }
  }

  Log::Info << "Neighbors computed." << endl;

  // Compute recall, if desired.
  if (CLI::HasParam("true_neighbors"))
  {
    // Load the true neighbors.
    arma::Mat<size_t> trueNeighbors =
        std::move(CLI::GetParam<arma::Mat<size_t>>("true_neighbors"));
    Log::Info << "Loaded true neighbor indices from '"
        << CLI::GetUnmappedParam<arma::Mat<size_t>>("true_neighbors") << "'."
        << endl;

    // Compute recall and print it.
    double recallPercentage = 100 * allkann.ComputeRecall(neighbors,
        trueNeighbors);

    Log::Info << "Recall: " << recallPercentage << endl;
  }

  // Save output, if desired.
  if (CLI::HasParam("distances"))
    CLI::GetParam<arma::mat>("distances") = std::move(distances);
  if (CLI::HasParam("neighbors"))
    CLI::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
  if (CLI::HasParam("output_model_file"))
    data::Save(outputModelFile, "lsh_model", allkann);
}
