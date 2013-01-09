/**
 * @file lsh_main.cpp
 * @author Parikshit Ram
 *
 * This file computes the approximate nearest-neighbors using 2-stable 
 * Locality-sensitive Hashing. 
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
    "This program will calculate the k approximate-nearest-neighbors "
    "of a set of points. You may specify a separate set of reference "
    "points and query points, or just a reference set which will be "
    "used as both the reference and query set. "
    "\n\n"
    "For example, the following will return 5 neighbors from the "
    "data for each point in 'input.csv' "
    "and store the distances in 'distances.csv' and the neighbors in the "
    "file 'neighbors.csv':"
    "\n\n"
    "$ ./lsh/lsh -k 5 -r input.csv -d distances.csv -n neighbors.csv "
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th nearest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.", "r");
PARAM_STRING("distances_file", "File to output distances into.", "d", "");
PARAM_STRING("neighbors_file", "File to output neighbors into.", "n", "");
PARAM_INT_REQ("k", "Number of nearest neighbors to find.", "k");
PARAM_STRING("query_file", "File containing query points (optional).", "q", "");

PARAM_INT("num_projections", "The number of hash functions for each table", 
          "K", 10);
PARAM_INT("num_tables", "The number of hash tables to be used.", "L", 30);
PARAM_DOUBLE("hash_width", "The hash width for the first-level hashing "
             "in the LSH preprocessing. By default, the LSH class "
             "automatically estimates a hash width for its use.", "H", 0.0);
PARAM_INT("second_hash_size", "The size of the second level hash table.", 
          "M", 99901);
PARAM_INT("bucket_size", "The size of a bucket in the second level hash.", 
          "B", 500);

int main(int argc, char *argv[])
{
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);
  math::RandomSeed(time(NULL)); 

  // Get all the parameters.
  string referenceFile = CLI::GetParam<string>("reference_file");
  string distancesFile = CLI::GetParam<string>("distances_file");
  string neighborsFile = CLI::GetParam<string>("neighbors_file");

  size_t k = CLI::GetParam<int>("k");
  size_t secondHashSize = CLI::GetParam<int>("second_hash_size");
  size_t bucketSize = CLI::GetParam<int>("bucket_size");

  arma::mat referenceData;
  arma::mat queryData; // So it doesn't go out of scope.
  data::Load(referenceFile.c_str(), referenceData, true);

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


  // Pick up the 'K' and the 'L' parameter for LSH
  size_t numProj = CLI::GetParam<int>("num_projections");
  size_t numTables = CLI::GetParam<int>("num_tables");
  
  // Compute the 'hash_width' parameter from LSH
  double hashWidth = CLI::GetParam<double>("hash_width");


  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (CLI::GetParam<string>("query_file") != "")
  {
    string queryFile = CLI::GetParam<string>("query_file");

    data::Load(queryFile.c_str(), queryData, true);
    Log::Info << "Loaded query data from '" << queryFile << "' ("
              << queryData.n_rows << " x " << queryData.n_cols << ")." << endl;
  }

  if (hashWidth == 0.0)
    Log::Info << "LSH with " << numProj << " projections(K) and " << 
      numTables << " tables(L) with default hash width." << endl;
  else
    Log::Info << "LSH with " << numProj << " projections(K) and " << 
      numTables << " tables(L) with hash width(r): " << hashWidth << endl;

  Timer::Start("hash_building");

  LSHSearch<>* allkann;

  if (CLI::GetParam<string>("query_file") != "")
    allkann = new LSHSearch<>(referenceData, queryData, numProj, numTables,
                              hashWidth, secondHashSize, bucketSize);
  else
    allkann = new LSHSearch<>(referenceData, numProj, numTables, hashWidth,
                              secondHashSize, bucketSize);

  Timer::Stop("hash_building");
  
  Log::Info << "Computing " << k << " distance approx. nearest neighbors " << 
    endl;
  allkann->Search(k, neighbors, distances);

  Log::Info << "Neighbors computed." << endl;

  // Save output.
  if (distancesFile != "") 
    data::Save(distancesFile, distances);

  if (neighborsFile != "")
    data::Save(neighborsFile, neighbors);

  delete allkann;

  return 0;
}
