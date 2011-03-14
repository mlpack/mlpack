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
#include "fastlib/fx/io.h"

using namespace std;
using namespace mlpack;
using namespace allnn;

int main (int argc, char *argv[]) {
	
  IO::parseCommandLine(argc, argv);
  arma::mat data_for_tree;

  string input_file = IO::getValue<string>("input_file");
  string output_file;

  if(IO::checkValue("output_file"))
    output_file = IO::getValue<string>("output_file");
  else
    output_file = "output.csv";
  
  NOTIFY("Loading file %s...", input_file.c_str());
  data::Load(input_file.c_str(), data_for_tree);

  AllNN allnn(data_for_tree, fx_root);
  
  arma::Col<index_t> resulting_neighbors_tree;
  arma::vec resulting_distances_tree;

  NOTIFY("Computing neighbors...");

  allnn.ComputeNeighbors(resulting_distances_tree, resulting_neighbors_tree);

  NOTIFY("Saving results to %s...", output_file.c_str());

  data::Save(output_file.c_str(), resulting_neighbors_tree, resulting_distances_tree); 
}
