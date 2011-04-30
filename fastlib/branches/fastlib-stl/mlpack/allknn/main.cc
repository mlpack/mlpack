/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/27/2008 11:52:43 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <fastlib/fastlib.h>
#include "allknn.h"
#include <fastlib/fx/io.h>

#include <string>
#include <errno.h>

#include <armadillo>

// Define our input parameters that this program will take.
PARAM_MODULE("allknn", "The all-k-nearest-neighbors computation.");
PARAM_STRING_REQ("reference_file", "CSV file containing the reference dataset.",
    "allknn");
PARAM_STRING("query_file", "CSV file containing query points (optional).",
    "allknn");
PARAM_STRING_REQ("output_file", "File to output CSV-formatted results into.",
    "allknn");
PARAM_INT("k", "Number of nearest neighbors to compute.", "allknn");
PARAM_INT("leaf_size", "Leaf size for kd-tree calculation.", "allknn");
PARAM_BOOL("single_mode", "If true, use single-tree mode (instead of " 
    "dual-tree).", "allknn");
PARAM_BOOL("naive_mode", "If true, use naive computation (no trees).  This "
    "overrides single_mode.", "allknn");

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
  int knn_options = (single_mode ? AllkNN::MODE_SINGLE : 0) |
      (naive_mode ? AllkNN::NAIVE : 0);

  arma::Col<index_t> neighbors;
  arma::vec distances;

  if (data::Load(reference_file.c_str(), reference_data) == SUCCESS_FAIL) {
    FATAL("Reference file %s not found", reference_file.c_str());
  }
  NOTIFY("Loaded reference data from file %s", reference_file.c_str());

  AllkNN* allknn = NULL;
 
  if (IO::CheckValue("query_file")) {
    std::string query_file = IO::GetValue<std::string>("query_file");
    arma::mat query_data;

    if (data::Load(query_file.c_str(), query_data) == SUCCESS_FAIL) {
      FATAL("Query file %s not found", query_file.c_str());
    }
    NOTIFY("Query data loaded from %s", query_file.c_str());

    NOTIFY("Building query and reference tree"); 
    allknn = new AllkNN(query_data, reference_data, (datanode*) NULL, knn_options);

  } else {
    NOTIFY("Building reference tree");
    allknn = new AllkNN(reference_data, (datanode*) NULL, knn_options);
  }

  NOTIFY("Tree(s) built");

  int k = IO::GetValue<int>("allknn/k");
  
  NOTIFY("Computing %"LI"d nearest neighbors", k);
  allknn->ComputeNeighbors(neighbors, distances);

  NOTIFY("Neighbors computed");
  NOTIFY("Exporting results");
  FILE *fp = fopen(output_file.c_str(), "w");
  if (fp == NULL) {
    FATAL("Error while opening %s...%s", output_file.c_str(),
        strerror(errno));
  }
  for(index_t i = 0; i < neighbors.n_elem / k; i++) {
    for(index_t j = 0; j < k; j++) {
      fprintf(fp, "%"LI"d %"LI"d %lg\n", i, neighbors[i * k + j], distances[i * k + j]);
    }
  }
  fclose(fp);

  delete allknn;
}
