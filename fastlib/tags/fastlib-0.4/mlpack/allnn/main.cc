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

using namespace std;

int main (int argc, char *argv[]) {
  fx_module *fx_root = fx_init(argc, argv, &allnn_doc);
  AllNN allnn;
  Matrix data_for_tree;

  string input_file = fx_param_str_req(fx_root, "input_file");
  string output_file = fx_param_str(fx_root, "output_file", "output.csv");

  NOTIFY("Loading file %s...", input_file.c_str());
  data::Load(input_file.c_str(), &data_for_tree);
  allnn.Init(data_for_tree, fx_root);
  
  GenVector<index_t> resulting_neighbors_tree;
  GenVector<double> resulting_distances_tree;

  NOTIFY("Computing neighbors...");

  allnn.ComputeNeighbors(&resulting_neighbors_tree, &resulting_distances_tree);

  NOTIFY("Saving results to %s...", output_file.c_str());

  data::Save(output_file.c_str(), resulting_neighbors_tree, resulting_distances_tree); 

  fx_done(fx_root);
}
