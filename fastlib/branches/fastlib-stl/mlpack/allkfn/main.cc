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


int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, NULL);
  std::string result_file = fx_param_str(module, "result_file", "result.txt");
  std::string reference_file = fx_param_str_req(module, "reference_file");
  Matrix reference_data;
  ArrayList<index_t> neighbors;
  ArrayList<double> distances;
  if (data::Load(reference_file.c_str(), &reference_data)==SUCCESS_FAIL) {
    FATAL("Reference file %s not found", reference_file.c_str());
  }
  NOTIFY("Loaded reference data from file %s", reference_file.c_str());
 
  AllkFN allkfn; 
  if (fx_param_exists(module, "query_file")) {
    std::string query_file=fx_param_str_req(module, "query_file");
    Matrix query_data;
    if (data::Load(query_file.c_str(), &query_data)==SUCCESS_FAIL) {
      FATAL("Query file %s not found", query_file.c_str());
    }
    NOTIFY("Query data loaded from %s", query_file.c_str());
    NOTIFY("Building query and reference tree"); 
    allkfn.Init(query_data, reference_data, module);
  } else {
    NOTIFY("Building reference tree");
    allkfn.Init(reference_data, module);
  }
  NOTIFY("Tree(s) built");
  index_t kfns=fx_param_int_req(module, "kfns");
  NOTIFY("Computing %"LI"d furthest neighbors", kfns);
  allkfn.ComputeNeighbors(&neighbors, &distances);
  NOTIFY("Neighbors computed");
  NOTIFY("Exporting results");
  FILE *fp=fopen(result_file.c_str(), "w");
  if (fp==NULL) {
    FATAL("Error while opening %s...%s", result_file.c_str(),
        strerror(errno));
  }
  for(index_t i=0 ; i < neighbors.size()/kfns ; i++) {
    for(index_t j=0; j<kfns; j++) {
      fprintf(fp, "%"LI"d %"LI"d %lg\n", i, neighbors[i*kfns+j], distances[i*kfns+j]);
    }
  }
  fclose(fp);
  fx_done(module);
}
