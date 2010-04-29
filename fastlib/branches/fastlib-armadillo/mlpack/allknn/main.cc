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
#include "allknn.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, NULL);
  std::string result_file = fx_param_str(module, "result_file", "result.txt");
  std::string reference_file = fx_param_str_req(module, "reference_file");
  Matrix reference_data;
  arma::mat tmp_ref;

  ArrayList<index_t> neighbors;
  ArrayList<double> distances;
  if (data::Load(reference_file.c_str(), tmp_ref)==SUCCESS_FAIL) {
    FATAL("Reference file %s not found", reference_file.c_str());
  }
  arma_compat::armaToMatrix(tmp_ref, reference_data);
  NOTIFY("Loaded reference data from file %s", reference_file.c_str());
 
  AllkNN allknn; 
  if (fx_param_exists(module, "query_file")) {
    std::string query_file=fx_param_str_req(module, "query_file");
    Matrix query_data;
    arma::mat tmp_query;
    if (data::Load(query_file.c_str(), tmp_query)==SUCCESS_FAIL) {
      FATAL("Query file %s not found", query_file.c_str());
    }
    arma_compat::armaToMatrix(tmp_query, query_data);
    NOTIFY("Query data loaded from %s", query_file.c_str());
    NOTIFY("Building query and reference tree"); 
    allknn.Init(query_data, reference_data, module);
  } else {
    NOTIFY("Building reference tree");
    allknn.Init(reference_data, module);
  }
  NOTIFY("Tree(s) built");
  index_t knns=fx_param_int_req(module, "knns");
  NOTIFY("Computing %"LI"d nearest neighbors", knns);
  allknn.ComputeNeighbors(&neighbors, &distances);
  NOTIFY("Neighbors computed");
  NOTIFY("Exporting results");
  FILE *fp=fopen(result_file.c_str(), "w");
  if (fp==NULL) {
    FATAL("Error while opening %s...%s", result_file.c_str(),
        strerror(errno));
  }
  for(index_t i=0 ; i < neighbors.size()/knns ; i++) {
    for(index_t j=0; j<knns; j++) {
      fprintf(fp, "%"LI"d %"LI"d %lg\n", i, neighbors[i*knns+j], distances[i*knns+j]);
    }
  }
  fclose(fp);
  fx_done(module);
}
