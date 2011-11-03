/**
 * @file main.cc
 *
 * Implementation of the AllkNN executable.  Allows some number of standard
 * options.
 *
 * @author Ryan Curtin
 */
#include <mlpack/core.h>
#include "neighbor_search.h"

#include <string>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;

// Information about the program itself.
PROGRAM_INFO("All K-Nearest-Neighbors",
    "This program will calculate the all k-nearest-neighbors of a set of "
    "points. You may specify a separate set of reference points and query "
    "points, or just a reference set which will be used as both the reference "
    "and query set."
    "\n\n"
    "For example, the following will calculate the 5 nearest neighbors of each"
    "point in 'input.csv' and store the results in 'output.csv':"
    "\n\n"
    "$ allknn --neighbor_search/k=5 --reference_file=input.csv\n"
    "  --output_file=output.csv", "neighbor_search");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "CSV file containing the reference dataset.",
    "");
PARAM_STRING("query_file", "CSV file containing query points (optional).",
    "", "");
PARAM_STRING_REQ("output_file", "File to output CSV-formatted results into.",
    "");

int main(int argc, char *argv[]) {
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  string reference_file = CLI::GetParam<string>("reference_file");
  string output_file = CLI::GetParam<string>("output_file");

  arma::mat reference_data;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (!data::Load(reference_file.c_str(), reference_data))
    Log::Fatal << "Reference file " << reference_file << " not found." << endl;

  Log::Info << "Loaded reference data from " << reference_file << endl;

  // Sanity check on k value: must be greater than 0, must be less than the
  // number of reference points.
  size_t k = CLI::GetParam<int>("neighbor_search/k");
  if ((k <= 0) || (k >= reference_data.n_cols)) {
    Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less ";
    Log::Fatal << "than the number of reference points (";
    Log::Fatal << reference_data.n_cols << ")." << endl;
  }

  // Sanity check on leaf size.
  if (CLI::GetParam<int>("tree/leaf_size") <= 0) {
    Log::Fatal << "Invalid leaf size: " << CLI::GetParam<int>("allknn/leaf_size")
        << endl;
  }

  AllkNN* allknn = NULL;

  if (CLI::GetParam<string>("query_file") != "") {
    string query_file = CLI::GetParam<string>("query_file");
    arma::mat query_data;

    if (!data::Load(query_file.c_str(), query_data))
      Log::Fatal << "Query file " << query_file << " not found" << endl;

    Log::Info << "Query data loaded from " << query_file << endl;

    Log::Info << "Building query and reference trees..." << endl;
    allknn = new AllkNN(query_data, reference_data);

  } else {
    Log::Info << "Building reference tree..." << endl;
    allknn = new AllkNN(reference_data);
  }

  Log::Info << "Tree(s) built." << endl;

  Log::Info << "Computing " << k << " nearest neighbors..." << endl;
  allknn->ComputeNeighbors(neighbors, distances);

  Log::Info << "Neighbors computed." << endl;
  Log::Info << "Exporting results..." << endl;

  // Should be using data::Save or a related function instead of being written
  // by hand.
  try {
    ofstream out(output_file.c_str());

    for (size_t col = 0; col < neighbors.n_cols; col++) {
      out << col << ", ";
      for (size_t j = 0; j < (k - 1) /* last is special case */; j++) {
        out << neighbors(j, col) << ", " << distances(j, col) << ", ";
      }
      out << neighbors((k - 1), col) << ", " << distances((k - 1), col) << endl;
    }

    out.close();
  } catch(exception& e) {
    Log::Fatal << "Error while opening " << output_file << ": " << e.what()
        << endl;
  }
  delete allknn;
}
