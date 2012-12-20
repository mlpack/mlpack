/**
 * @file lsh_analysis_main.cpp
 * @author Parikshit Ram
 *
 * This main file computes the accuracy-time tradeoff of 
 * the 2-stable LSH class 'LSHSearch'
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
PROGRAM_INFO("LSH accuracy-time tradeoff analysis.",
    "This program will calculate the k approximate-nearest-neighbors "
    "of a set of queries under different parameter settings."
    " You may specify a separate set of reference "
    "points and query points, or just a reference set which will be "
    "used as both the reference and query set. "
    "\n\n"
    "For a given set of queries and references and 'k', this program "
    "uses different parameters for LSH and computes the error and "
    "the query time. For the computation of error, it requires the "
    "files containing the ranks and the NN distances for each query. "
    "\n\n"
    "Sample usage: \n"
    "./lsh/lsh_analysis -r reference.csv -q queries.csv -k 5 "
    " -P -W -E query_ranks_file.csv -D query_nn_dist_file.csv "
    " -F error_v_time_output.csv ");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
                 "r");
PARAM_INT_REQ("k", "Number of nearest neighbors to find.", "k");
PARAM_STRING("query_file", "File containing query points (optional).", "q", "");

PARAM_STRING_REQ("rank_file", "The file containing the true ranks.", "E");
PARAM_STRING_REQ("nn_dist_file", "The file containing the true distance"
                 " errors.", "D");
PARAM_INT("num_projections", "The number of hash functions for each table",
          "K", 10);
PARAM_INT("num_tables", "The number of hash tables to be used.", "L", 30);
PARAM_INT("second_hash_size", "The size of the second level hash table.", 
          "M", 8807);
PARAM_INT("bucket_size", "The size of a bucket in the second level hash.", 
          "B", 500);

PARAM_FLAG("try_diff_params", "The flag to trigger the search with "
           "different 'K', 'L' and 'r'.", "P");
PARAM_FLAG("try_diff_widths", "The flag to trigger the search with "
           "different hash widths.", "W");

PARAM_STRING("error_time_file", "File to output the error v. time report to.",
             "F", "");

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

  Log::Info << "Loaded reference data from '" << referenceFile << "' (" << 
    referenceData.n_rows << " x " << referenceData.n_cols << ")." << endl;

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
    numProjs.set_size(4);
    numProjs << 10 << 25 << 40 << 55;
    numTables.set_size(4);
    numTables << 5 << 10 << 15 << 20;
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
    size_t p1 = (size_t) math::RandInt(referenceData.n_cols);
    size_t p2 = (size_t) math::RandInt(referenceData.n_cols);
    avgDist += metric::EuclideanDistance::Evaluate(referenceData.unsafe_col(p1),
                                                   referenceData.unsafe_col(p2));
  }

  avgDist /= 25;

  Log::Info << "Hash width chosen as: " << avgDist << endl;

  arma::vec hashWidths;
  if (tryDiffWidths)
  {
    arma::vec eps(3);
    eps << 0.01 << 0.1 << 1.0;
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
  arma::mat allNeighborDistances;
  
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (CLI::GetParam<string>("query_file") != "")
  {
    string queryFile = CLI::GetParam<string>("query_file");

    data::Load(queryFile.c_str(), queryData, true);
    Log::Info << "Loaded query data from '" << queryFile << "' (" << 
      queryData.n_rows << " x " << queryData.n_cols << ")." << endl;

    allNeighbors.set_size(k * timesTaken.n_elem, queryData.n_cols);
    allNeighborDistances.set_size(k * timesTaken.n_elem, queryData.n_cols);
  }
  else
  {
    allNeighbors.set_size(k * timesTaken.n_elem, referenceData.n_cols);
    allNeighborDistances.set_size(k * timesTaken.n_elem, referenceData.n_cols);
  }

  size_t exptInd = 0;
  arma::mat exptParams(timesTaken.n_elem, 3);

  // Looping through all combinations of the parameters and noting the 
  // runtime and the results.
  for (size_t widthInd = 0; widthInd < hashWidths.n_elem; widthInd++)
  {
    for (size_t projInd = 0; projInd < numProjs.n_elem; projInd++)
    {
      for (size_t tableInd = 0; tableInd < numTables.n_elem; tableInd++)
      {
        Log::Info << "LSH with K: " << numProjs[projInd] << ", L: " << 
          numTables[tableInd] << ", r: " << hashWidths[widthInd] << endl;

        Timer::Start("hash_building");

        LSHSearch<>* allkann;

        if (CLI::GetParam<string>("query_file") != "")
          allkann = new LSHSearch<>(referenceData, queryData, numProjs[projInd],
                                    numTables[tableInd], hashWidths[widthInd],
                                    secondHashSize, bucketSize);
        else
          allkann = new LSHSearch<>(referenceData, numProjs[projInd],
                                    numTables[tableInd], hashWidths[widthInd],
                                    secondHashSize, bucketSize);

        Timer::Stop("hash_building");

        timeval start_tv = Timer::Get("computing_neighbors");
        double startTime 
          = (double) start_tv.tv_sec + (double) start_tv.tv_usec / 1.0e6;

        Log::Info << "Computing " << k << " approx. nearest neighbors" << endl;
        allkann->Search(k, neighbors, distances);
        Log::Info << "Neighbors computed." << endl;

        timeval stop_tv = Timer::Get("computing_neighbors");
        double stopTime 
          = (double) stop_tv.tv_sec + (double) stop_tv.tv_usec / 1.0e6;

        exptParams(exptInd, 0) = (double) numProjs[projInd];
        exptParams(exptInd, 1) = (double) numTables[tableInd];
        exptParams(exptInd, 2) = hashWidths[widthInd];
        timesTaken[exptInd] = stopTime - startTime;

        // add results to big matrix
        allNeighbors.rows(exptInd * k, (exptInd + 1) * k - 1) = neighbors;
        allNeighborDistances.rows(exptInd * k, (exptInd + 1) * k - 1) 
          = distances;

        exptInd++;

        neighbors.reset();
        distances.reset();

        delete allkann;
      } // diff. L
    } // diff. K
  } // diff. 'width'


  // Computing the errors
  string rankFile = CLI::GetParam<string>("rank_file");
  Log::Warn << "Computing error..." << endl;

  contrib_utils::LineReader lr(rankFile);

  arma::mat allANNErrors(timesTaken.n_elem, 8);
  // 0 - K
  // 1 - L
  // 2 - width
  // 3 - Time taken
  // 4 - Mean Rank/Recall
  // 5 - Median Rank/Recall
  // 6 - StdDev Rank/Recall
  // 7 - MaxRank / MinRecall


  // If k == 1, compute the rank and distance errors
  if (k == 1) 
  {
    string distFile = CLI::GetParam<string>("nn_dist_file");
    arma::mat nnDistsMat;

    if (!data::Load(distFile, nnDistsMat))
      Log::Fatal << "Dist file " << distFile << " cannot be loaded." << endl;

    arma::vec nnDists(nnDistsMat.row(0).t());

    allANNErrors.resize(timesTaken.n_elem, 12);
    // 8 - Mean DE
    // 9 - Median DE
    // 10 - StdDev DE
    // 11 - Max DE

    arma::mat ranks(timesTaken.n_elem, allNeighbors.n_cols);
    arma::mat des(timesTaken.n_elem, allNeighbors.n_cols);

    for (size_t i = 0; i < allNeighbors.n_cols; i++)
    {
      arma::Col<size_t> true_ranks(referenceData.n_cols);
      lr.ReadLine(&true_ranks);

      for (size_t j = 0; j < timesTaken.n_elem; j++) 
      {
        if (allNeighbors(j, i) < referenceData.n_cols)
        {
          ranks(j, i) = (double) true_ranks[allNeighbors(j, i)];
          des(j, i) = (allNeighborDistances(j, i) - nnDists[i]) / nnDists[i];
        }
        // not sure what to do in terms of distance error 
        // in case no result is returned
        else 
        {
          ranks(j, i) = (double) referenceData.n_cols;
          des(j, i) = 1000;
        }
      }
    }

    // Saving the LSH parameters and the query times
    allANNErrors.cols(0, 2) = exptParams;
    allANNErrors.col(3) = timesTaken;

    // Saving the mean distance error and rank over all queries.
    allANNErrors.col(4) = arma::mean(ranks, 1);
    allANNErrors.col(5) = arma::median(ranks, 1);
    allANNErrors.col(6) = arma::stddev(ranks, 1, 1);
    allANNErrors.col(7) = arma::max(ranks, 1);

    allANNErrors.col(8) = arma::mean(des, 1);
    allANNErrors.col(9) = arma::median(des, 1);
    allANNErrors.col(10) = arma::stddev(des, 1, 1);
    allANNErrors.col(11) = arma::max(des, 1);

  } // if k == 1
  // if k > 1, compute the recall of the k-nearest-neighbors.
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
        {
          if (allNeighbors(j * k + ind, i) < referenceData.n_cols)
          {
            if (true_ranks[allNeighbors(j * k + ind, i)] <= k)
              recalls(j, i)++;
          }
        }
      }
    }

    recalls /= k;
    
    // Saving the LSH parameters and the query times
    allANNErrors.cols(0, 2) = exptParams;
    allANNErrors.col(3) = timesTaken;

    // Saving the mean recall of the k-nearest-neighbor over all queries.
    allANNErrors.col(4) = arma::mean(recalls, 1);
    allANNErrors.col(5) = arma::median(recalls, 1);
    allANNErrors.col(6) = arma::stddev(recalls, 1, 1);
    allANNErrors.col(7) = arma::min(recalls, 1);
    
  } // if k > 1, compute recall of k-NN

  Log::Warn << allANNErrors;


  // Saving the output in a file
  string annErrorOutputFile = CLI::GetParam<string>("error_time_file");

  if (annErrorOutputFile != "")
  {    
    allANNErrors = allANNErrors.t();
    data::Save(annErrorOutputFile, allANNErrors);
  }    
  else 
  {
    Log::Warn << "Params: " << endl << exptParams.t() << "Times Taken: " <<
      endl << timesTaken.t();
  }

  return 0;
}
