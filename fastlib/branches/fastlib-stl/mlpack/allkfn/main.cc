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
#include "allkfn.h"

#include <string>
#include <errno.h>

#include <armadillo>
#include <fastlib/base/arma_compat.h>

using namespace mlpack;
using namespace mlpack::allkfn;

int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, NULL);
  std::string result_file = fx_param_str(module, "result_file", "result.txt");
  std::string reference_file = fx_param_str_req(module, "reference_file");
  arma::mat reference_data;

  bool single_mode = (fx_param_int(module, "single_mode", 0) == 1) ? true :
      false;

  arma::Col<index_t> neighbors;
  arma::vec distances;
  if (data::Load(reference_file.c_str(), reference_data) == SUCCESS_FAIL) {
    IO::Fatal << "Reference file " << reference_file.c_str() << " not found" << std::endl;
  }
  IO::Info << "Loaded reference data from file " << reference_file.c_str() << std:: endl;
 
  AllkFN* allkfn = NULL;
 
  if (fx_param_exists(module, "query_file")) {
    std::string query_file = fx_param_str_req(module, "query_file");
    arma::mat query_data;
    if (data::Load(query_file.c_str(), query_data) == SUCCESS_FAIL) {
      IO::Fatal << "Query file " << query_file.c_str() << " not found" << std::endl;
    }
    IO::Info << "Query data loaded from " << query_file.c_str() << std::endl; 
    IO::Info << "Building query and reference tree" << std::endl;
    allkfn = new AllkFN(query_data, reference_data, module,
        single_mode ? AllkFN::MODE_SINGLE : 0);
  } else {
    IO::Info << "Building reference tree" << std::endl; 
    allkfn = new AllkFN(reference_data, module,
        single_mode ? AllkFN::MODE_SINGLE : 0);
  }
  IO::Info << "Tree(s) built" << std::endl;
  index_t kfns = fx_param_int_req(module, "kfns");
  IO::Info << "Computing " << kfns << LI << "d furthest neighbors" << std::endl;
  allkfn->ComputeNeighbors(neighbors, distances);
  IO::Info << "Neighbors computed " << std::endl;
  IO::Info << "Exporting results " << std::endl;
  FILE *fp = fopen(result_file.c_str(), "w");
  if (fp == NULL) {
    IO::Fatal << "Error while opening " << result_file.c_str() 
	      << "..." << strerror(errno) << std::endl; 
  }
  for (index_t i = 0; i < neighbors.n_elem / kfns; i++) {
    for (index_t j = 0; j < kfns; j++) {
      fprintf(fp, "%"LI"d %"LI"d %lg\n", i, neighbors[i * kfns + j], distances[i * kfns + j]);
    }
  }
  fclose(fp);

  delete allkfn;

  fx_done(module);
}
