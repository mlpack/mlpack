/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/14/2008 07:15:55 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <string>
#include "allnn.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

int main (int argc, char *argv[]) {
  fx_module *fx_root=fx_init(argc, argv, NULL);
  AllNN allnn;
  Matrix data_for_tree;
  arma::mat tmp_data;
  
  std::string filename=fx_param_str_req(fx_root, "file");
  NOTIFY("Loading file...");
  data::Load(filename.c_str(), tmp_data);
  arma_compat::armaToMatrix(tmp_data, data_for_tree);
  NOTIFY("File loaded...");
  allnn.Init(data_for_tree, fx_root);
  //GenVector<index_t> resulting_neighbors_tree;
  //GenVector<double> resulting_distances_tree;
  NOTIFY("Computing Neighbors...");
  allnn.ComputeNeighbors(NULL, NULL);
  NOTIFY("Neighbors Computed...");
  fx_done(fx_root);
}
