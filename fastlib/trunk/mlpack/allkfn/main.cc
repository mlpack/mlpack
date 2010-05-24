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

#include <string>
#include <errno.h>
#include "fastlib/fastlib.h"
#include "allkfn.h"

const fx_entry_doc allkfn_entries[] = {
  {"input_file", FX_REQUIRED, FX_STR, NULL,
   "Input dataset (CSV or ARFF) to read from\n"},
  {"query_file", FX_PARAM, FX_STR, NULL,
   "Dataset (CSV or ARFF) containing query points; if omitted, all of the\n"
   "   input dataset points will be queried.\n"},
  {"output_file", FX_PARAM, FX_STR, NULL,
   "File to output results into (default output.csv)\n"},
  {"k", FX_REQUIRED, FX_INT, NULL,
   "Number of furthest neighbors to compute\n"},
  {"leaf_size", FX_PARAM, FX_INT, NULL,
   "Maximum number of points to store in a leaf (default 20).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc allkfn_doc = {
  allkfn_entries, NULL,
  "This is the MLPACK implementation of dual-tree all-furthest-neighbors.  It\n"
  "accepts an input dataset as a parameter, and optionally a set of query\n"
  "points; then, it calculates the k furthest neighbors from each of the query\n"
  "points.  If the query point dataset is not specified, the k furthest\n"
  "neighbors of each point in the input dataset are calculated.\n"
  "\n"
  "Output is stored in a CSV file which is specified as a parameter (default\n"
  "is 'output.csv') in the following format, for each of the query points:\n"
  "  <index of query point>, <index of furthest neighbor>,     <distance>\n"
  "  <index of query point>, <index of 2nd furthest neighbor>, <distance>\n"
  "  ...\n"
  "  <index of query point>, <index of kth furthest neighbor>, <distance>\n"
  "\n"
  "The given distance is squared L2 distance.\n"
};

int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, &allkfn_doc);
  std::string result_file = fx_param_str(module, "output_file", "output.csv");
  std::string reference_file = fx_param_str_req(module, "input_file");
  Matrix reference_data;
  ArrayList<index_t> neighbors;
  ArrayList<double> distances;
  if (data::Load(reference_file.c_str(), &reference_data)==SUCCESS_FAIL) {
    FATAL("Reference file %s not found!", reference_file.c_str());
  }
  NOTIFY("Loaded reference data from file %s.", reference_file.c_str());
 
  AllkFN allkfn; 
  if (fx_param_exists(module, "query_file")) {
    std::string query_file=fx_param_str_req(module, "query_file");
    Matrix query_data;
    if (data::Load(query_file.c_str(), &query_data)==SUCCESS_FAIL) {
      FATAL("Query file %s not found!", query_file.c_str());
    }
    NOTIFY("Query data loaded from %s.", query_file.c_str());
    NOTIFY("Building query and reference tree..."); 
    allkfn.Init(query_data, reference_data, module);
  } else {
    NOTIFY("Building reference tree...");
    allkfn.Init(reference_data, module);
  }
  index_t kfns=fx_param_int_req(module, "k");
  NOTIFY("Computing %"LI"d furthest neighbors...", kfns);
  allkfn.ComputeNeighbors(&neighbors, &distances);
  NOTIFY("Writing results to %s...", result_file.c_str());
  FILE *fp = fopen(result_file.c_str(), "w");
  if (fp == NULL) {
    FATAL("Error while opening %s...%s", result_file.c_str(),
        strerror(errno));
  }
  for(index_t i=0 ; i < neighbors.size()/kfns ; i++) {
    for(index_t j=0; j<kfns; j++) {
      fprintf(fp, "%"LI"d, %"LI"d, %lg\n", i, neighbors[i*kfns+j], distances[i*kfns+j]);
    }
  }
  fclose(fp);
  fx_done(module);
}
