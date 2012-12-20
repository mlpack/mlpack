/**
 * @file lsh_main.cpp
 * @author Parikshit Ram
 *
 * Implementation of LSH with a 2-stable distribution of 
 * nearest neighbor search in Euclidean space. 
 */
#include <time.h>

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <string>
#include <fstream>
#include <iostream>

#include "lsh_search.hpp"
#include "../utils/utils.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;

// Information about the program itself.
PROGRAM_INFO("All K-distance-Approximate-Nearest-Neighbors with LSH",
    "This program will calculate the k distance-approximate-nearest-neighbors "
    "of a set of points. You may specify a separate set of reference "
    "points and query points, or just a reference set which will be "
    "used as both the reference and query set. "
    "\n\n"
    "For example, the following will return 5 neighbors from the "
    "data for each point in 'input.csv' "
    "and store the distances in 'distances.csv' and the neighbors in the "
    "file 'neighbors.csv':"
    "\n\n"
    "$ allkdann --k=5 --reference_file=input.csv --distances_file=distances.csv\n"
    "  --neighbors_file=neighbors.csv"
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

PARAM_INT("num_projections", "The number of hash functions for each table", "K", 10);
PARAM_INT("num_tables", "The number of hash tables to be used.", "L", 30);
PARAM_INT("second_hash_size", "The size of the second level hash table.", "M", 8807);
PARAM_INT("bucket_size", "The size of a bucket in the second level hash.", "B", 500);

PARAM_FLAG("try_diff_params", "The flag to trigger the search with "
           "different 'K', 'L' and 'r'.", "P");
PARAM_FLAG("try_diff_widths", "The flag to trigger the search with "
           "different hash widths.", "W");

PARAM_STRING("rank_file", "The file containing the true ranks.", "E", "");
PARAM_STRING("de_file", "The file containing the true distance errors.", "D", "");
PARAM_STRING("ann_error_file", "File to output the RANN errors to.", "F", "");

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

  bool tryDiffParams = CLI::HasParam("try_diff_params");
  bool tryDiffWidths = CLI::HasParam("try_diff_widths");

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
  arma::Col<size_t> numProjs, numTables;
  if (tryDiffParams) 
  {
    numProjs.set_size(3);
    numProjs << 10 << 25 << 50;
    numTables.set_size(5);
    numTables << 5 << 10 << 25 << 50 << 100;
  }
  else 
  {
    numProjs.set_size(1);
    numProjs[0] = CLI::GetParam<int>("num_projections");
    numTables.set_size(1);
    numTables[0] = CLI::GetParam<int>("num_tables");
  }

  // Compute the 'width' parameter from LSH

  // Find the average pairwise distance of 25 random pairs
  double avgDist = 0;
  for (size_t i = 0; i < 25; i++)
  {
    size_t p1 = (size_t) math::RandInt(referenceData.n_cols),
      p2 = (size_t) math::RandInt(referenceData.n_cols);

    avgDist += metric::EuclideanDistance::Evaluate(referenceData.unsafe_col(p1),
                                                   referenceData.unsafe_col(p2));
  }

  avgDist /= 25;

  Log::Info << "Hash width chosen as: " << avgDist << endl;

  arma::vec hashWidths;
  if (tryDiffWidths)
  {
    arma::vec eps(5);
    eps << 0.001 << 0.01 << 0.1 << 1.0 << 10.0;
    hashWidths = avgDist * eps;

  }
  else 
  {
    hashWidths.set_size(1);
    hashWidths[0] = avgDist;
  }

  arma::vec timesTaken(numProjs.n_elem * numTables.n_elem * hashWidths.n_elem);
  timesTaken.zeros();

  arma::Mat<size_t> allNeighbors;
  
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (CLI::GetParam<string>("query_file") != "")
  {
    string queryFile = CLI::GetParam<string>("query_file");

    data::Load(queryFile.c_str(), queryData, true);
    Log::Info << "Loaded query data from '" << queryFile << "' ("
              << queryData.n_rows << " x " << queryData.n_cols << ")." << endl;

    allNeighbors.set_size(k * timesTaken.n_elem, queryData.n_cols);
  }
  else
    allNeighbors.set_size(k * timesTaken.n_elem, referenceData.n_cols);

  size_t exptInd = 0;
  arma::mat exptParams(timesTaken.n_elem, 3);

  for (size_t widthInd = 0; widthInd < hashWidths.n_elem; widthInd++)
  {
    for (size_t projInd = 0; projInd < numProjs.n_elem; projInd++)
    {
      for (size_t tableInd = 0; tableInd < numTables.n_elem; tableInd++)
      {
        Log::Info << "LSH with K: " << numProjs[projInd] << ", L: " 
                  << numTables[tableInd] << ", r: " << hashWidths[widthInd] << endl;

        Timer::Start("hash_building");

        LSHSearch<>* allkdann;

        if (CLI::GetParam<string>("query_file") != "")
          allkdann = new LSHSearch<>(referenceData, queryData, numProjs[projInd],
                                     numTables[tableInd], hashWidths[widthInd],
                                     secondHashSize, bucketSize);
        else
          allkdann = new LSHSearch<>(referenceData, numProjs[projInd],
                                     numTables[tableInd], hashWidths[widthInd],
                                     secondHashSize, bucketSize);

        Timer::Stop("hash_building");

        timeval start_tv = Timer::Get("computing_neighbors");
        double startTime = (double) start_tv.tv_sec + (double) start_tv.tv_usec / 1.0e6;


        Log::Info << "Computing " << k << " distance approx. nearest neighbors " << endl;
        allkdann->Search(k, neighbors, distances);

        Log::Info << "Neighbors computed." << endl;

        timeval stop_tv = Timer::Get("computing_neighbors");
        double stopTime = (double) stop_tv.tv_sec + (double) stop_tv.tv_usec / 1.0e6;


        exptParams(exptInd, 0) = (double) numProjs[projInd];
        exptParams(exptInd, 1) = (double) numTables[tableInd];
        exptParams(exptInd, 2) = hashWidths[widthInd];
        timesTaken[exptInd] = stopTime - startTime;

        // add results to big matrix
        allNeighbors.rows(exptInd * k, (exptInd + 1) * k - 1) = neighbors;

        exptInd++;

        neighbors.reset();
        distances.reset();

        delete allkdann;

      } // diff. L
    } // diff. K
  } // diff. 'width'


  // TO FIX: Have to fix this since these matrices are getting reset.
  // Save output.
  if (distancesFile != "") 
    data::Save(distancesFile, distances);

  if (neighborsFile != "")
    data::Save(neighborsFile, neighbors);


  // Compute the error if the error file is provided
  string rankFile = CLI::GetParam<string>("rank_file");
  if (rankFile != "")
  {
    Log::Warn << "Computing error..." << endl;

    contrib_utils::LineReader lr(rankFile);

    arma::mat allDANNErrors(timesTaken.n_elem, 8);
    // 0 - K
    // 1 - L
    // 2 - width
    // 3 - Time taken
    // 4 - Mean Rank/Recall
    // 5 - Median Rank/Recall
    // 6 - StdDev Rank/Recall
    // 7 - MaxRank / MinRecall

    if (k == 1) 
    {
      string deFile = CLI::GetParam<string>("de_file");

      contrib_utils::LineReader *de_lr = NULL;

      if (deFile != "")
      {
        de_lr = new contrib_utils::LineReader(deFile);
        allDANNErrors.resize(timesTaken.n_elem, 12);
        // 8 - Mean DE
        // 9 - Median DE
        // 10 - StdDev DE
        // 11 - Max DE
      }

      arma::mat ranks(timesTaken.n_elem, allNeighbors.n_cols);
      arma::mat des(timesTaken.n_elem, allNeighbors.n_cols);

      for (size_t i = 0; i < allNeighbors.n_cols; i++)
      {
        arma::Col<size_t> true_ranks(referenceData.n_cols);
        lr.ReadLine(&true_ranks);

        arma::vec true_des(referenceData.n_cols);
        if (de_lr != NULL)
          de_lr->ReadLine(&true_des);

        for (size_t j = 0; j < timesTaken.n_elem; j++) 
        {
          if (allNeighbors(j, i) < referenceData.n_cols)
          {
            ranks(j, i) = (double) true_ranks[allNeighbors(j, i)];

            if (de_lr != NULL)
              des(j, i) = true_des[allNeighbors(j, i)];
          }
          else 
          {
            ranks(j, i) = (double) referenceData.n_cols;

            if (de_lr != NULL)
              des(j, i) = arma::max(true_des);
          }
        }
      }

      allDANNErrors.cols(0, 2) = exptParams;
      allDANNErrors.col(3) = timesTaken;
      allDANNErrors.col(4) = arma::mean(ranks, 1);
      allDANNErrors.col(5) = arma::median(ranks, 1);
      allDANNErrors.col(6) = arma::stddev(ranks, 1, 1);
      allDANNErrors.col(7) = arma::max(ranks, 1);

      if (de_lr != NULL) 
      {
        allDANNErrors.col(8) = arma::mean(des, 1);
        allDANNErrors.col(9) = arma::median(des, 1);
        allDANNErrors.col(10) = arma::stddev(des, 1, 1);
        allDANNErrors.col(11) = arma::max(des, 1);

        delete de_lr;
      }
    } // if k == 1, compute rank error and distance error
    else
    {
      arma::mat recalls(timesTaken.n_elem, allNeighbors.n_cols);
      recalls.zeros();

      for (size_t i = 0; i < allNeighbors.n_cols; i++)
      {
        arma::Col<size_t> true_ranks(referenceData.n_cols);
        lr.ReadLine(&true_ranks);

        for (size_t j = 0; j < timesTaken.n_elem; j++)
        {
          for (size_t ind = 0; ind < k; ind++)
            if (allNeighbors(j * k + ind, i) < referenceData.n_cols)
            {
              if (true_ranks[allNeighbors(j * k + ind, i)] <= k)
                recalls(j, i)++;
            }
          
        }
      }

      recalls /= k;
    
      allDANNErrors.cols(0, 2) = exptParams;
      allDANNErrors.col(3) = timesTaken;
      allDANNErrors.col(4) = arma::mean(recalls, 1);
      allDANNErrors.col(5) = arma::median(recalls, 1);
      allDANNErrors.col(6) = arma::stddev(recalls, 1, 1);
      allDANNErrors.col(7) = arma::min(recalls, 1);
    
    } // if k > 1, compute recall of k-NN

    Log::Warn << allDANNErrors;

    string annErrorOutputFile = CLI::GetParam<string>("ann_error_file");

    if (annErrorOutputFile != "")
    {    
      allDANNErrors = allDANNErrors.t();
      data::Save(annErrorOutputFile, allDANNErrors);
    }    
  }
  else 
  {
    Log::Warn << "Params: " << endl << exptParams.t()
              << "Times Taken: "  << endl << timesTaken.t();

  }

  return 0;
}
