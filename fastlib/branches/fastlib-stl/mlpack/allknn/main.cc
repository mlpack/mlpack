/**
 * @file main.cc
 *
 * Implementation of the AllkNN executable.  Allows some number of standard
 * options.
 *
 * @author Ryan Curtin
 */
#include <fastlib/fastlib.h>
#include "allknn.h"
#include <fastlib/fx/io.h>

#include <string>
#include <errno.h>

#include <armadillo>

using namespace std;

// Define our input parameters that this program will take.
PARAM_MODULE("allknn", "The all-k-nearest-neighbors computation.");
PARAM_STRING_REQ("reference_file", "CSV file containing the reference dataset.",
    "allknn");
PARAM_STRING("query_file", "CSV file containing query points (optional).",
    "allknn", "");
PARAM_STRING_REQ("output_file", "File to output CSV-formatted results into.",
    "allknn");
PARAM_INT("k", "Number of nearest neighbors to compute.", "allknn", 5);
PARAM_INT("leaf_size", "Leaf size for kd-tree calculation.", "allknn", 20);
PARAM_BOOL("single_mode", "If true, use single-tree mode (instead of " 
    "dual-tree).", "allknn", false);
PARAM_BOOL("naive_mode", "If true, use naive computation (no trees).  This "
    "overrides single_mode.", "allknn", false);

using namespace mlpack;
using namespace mlpack::allknn;

int main(int argc, char *argv[]) {
  // Give IO the command line parameters the user passed in.
  IO::ParseCommandLine(argc, argv);

  std::string reference_file =
      IO::GetValue<std::string>("allknn/reference_file");
  std::string output_file = IO::GetValue<std::string>("allknn/output_file");

  arma::mat reference_data;

  bool single_mode = IO::GetValue<bool>("allknn/single_mode");
  bool naive_mode = IO::GetValue<bool>("allknn/naive_mode");
  index_t knn_options = (single_mode ? AllkNN::MODE_SINGLE : 0) |
      (naive_mode ? AllkNN::NAIVE : 0);

  arma::Col<index_t> neighbors;
  arma::vec distances;

  if (data::Load(reference_file.c_str(), reference_data) == SUCCESS_FAIL)
    IO::Fatal << "Reference file " << reference_file << "not found." << endl;

  IO::Info << "Loaded reference data from " << reference_file << endl;

  AllkNN* allknn = NULL;
 
  if (IO::CheckValue("query_file")) {
    std::string query_file = IO::GetValue<std::string>("query_file");
    arma::mat query_data;

    if (data::Load(query_file.c_str(), query_data) == SUCCESS_FAIL)
      IO::Fatal << "Query file " << query_file << " not found" << endl;
    
    IO::Info << "Query data loaded from " << query_file << endl;

    IO::Info << "Building query and reference trees..." << endl;
    allknn = new AllkNN(query_data, reference_data, (datanode*) NULL,
        knn_options);

  } else {
    IO::Info << "Building reference tree..." << endl;
    allknn = new AllkNN(reference_data, (datanode*) NULL, knn_options);
  }

  IO::Info << "Tree(s) built." << endl;

  index_t k = IO::GetValue<index_t>("allknn/k");
  
  IO::Info << "Computing " << k << " nearest neighbors..." << endl;
  allknn->ComputeNeighbors(neighbors, distances);

  IO::Info << "Neighbors computed." << endl;
  IO::Info << "Exporting results..." << endl;

  // Should be using data::Save or a related function instead of being written
  // by hand.
  FILE *fp = fopen(output_file.c_str(), "w");
  if (fp == NULL) {
    IO::Fatal << "Error while opening " << output_file << ": " <<
        strerror(errno) << endl;
  }

  for(index_t i = 0; i < neighbors.n_elem / k; i++) {
    for(index_t j = 0; j < k; j++) {
      fprintf(fp, "%"LI"d %"LI"d %lg\n", i, neighbors[i * k + j], distances[i * k + j]);
    }
  }
  fclose(fp);

  delete allknn;
}
