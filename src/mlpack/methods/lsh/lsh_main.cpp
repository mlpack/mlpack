/**
 * @file lsh_main.cpp
 * @author Parikshit Ram
 *
 * This file computes the approximate nearest-neighbors using 2-stable
 * Locality-sensitive Hashing.
 *
 * This file is part of MLPACK 1.0.8.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <time.h>

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <string>
#include <fstream>
#include <iostream>

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
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
    "r");
PARAM_STRING("distances_file", "File to output distances into.", "d", "");
PARAM_STRING("neighbors_file", "File to output neighbors into.", "n", "");

PARAM_INT_REQ("k", "Number of nearest neighbors to find.", "k");

PARAM_STRING("query_file", "File containing query points (optional).", "q", "");

PARAM_INT("projections", "The number of hash functions for each table", "K",
    10);
PARAM_INT("tables", "The number of hash tables to be used.", "L", 30);
PARAM_DOUBLE("hash_width", "The hash width for the first-level hashing in the "
    "LSH preprocessing. By default, the LSH class automatically estimates a "
    "hash width for its use.", "H", 0.0);
PARAM_INT("second_hash_size", "The size of the second level hash table.", "M",
    99901);
PARAM_INT("bucket_size", "The size of a bucket in the second level hash.", "B",
    500);
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) time(NULL));

  // Get all the parameters.
  string referenceFile = CLI::GetParam<string>("reference_file");
  string distancesFile = CLI::GetParam<string>("distances_file");
  string neighborsFile = CLI::GetParam<string>("neighbors_file");

  size_t k = CLI::GetParam<int>("k");
  size_t secondHashSize = CLI::GetParam<int>("second_hash_size");
  size_t bucketSize = CLI::GetParam<int>("bucket_size");

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

  // Pick up the LSH-specific parameters.
  const size_t numProj = CLI::GetParam<int>("projections");
  const size_t numTables = CLI::GetParam<int>("tables");
  const double hashWidth = CLI::GetParam<double>("hash_width");

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (CLI::GetParam<string>("query_file") != "")
  {
    string queryFile = CLI::GetParam<string>("query_file");

    data::Load(queryFile, queryData, true);
    Log::Info << "Loaded query data from '" << queryFile << "' ("
              << queryData.n_rows << " x " << queryData.n_cols << ")." << endl;
  }

  if (hashWidth == 0.0)
    Log::Info << "Using LSH with " << numProj << " projections (K) and " <<
        numTables << " tables (L) with default hash width." << endl;
  else
    Log::Info << "Using LSH with " << numProj << " projections (K) and " <<
        numTables << " tables (L) with hash width(r): " << hashWidth << endl;

  Timer::Start("hash_building");

  LSHSearch<>* allkann;

  if (CLI::GetParam<string>("query_file") != "")
    allkann = new LSHSearch<>(referenceData, queryData, numProj, numTables,
                              hashWidth, secondHashSize, bucketSize);
  else
    allkann = new LSHSearch<>(referenceData, numProj, numTables, hashWidth,
                              secondHashSize, bucketSize);

  Timer::Stop("hash_building");

  Log::Info << "Computing " << k << " distance approximate nearest neighbors "
      << endl;
  allkann->Search(k, neighbors, distances);

  Log::Info << "Neighbors computed." << endl;

  // Save output.
  if (distancesFile != "")
    data::Save(distancesFile, distances);

  if (neighborsFile != "")
    data::Save(neighborsFile, neighbors);

  delete allkann;
}
