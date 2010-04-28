/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
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

#include <cerrno>
#include <string>
#include "fastlib/fastlib.h"
#include "allkfn.h"


int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, NULL);
  std::string result_file;
  std::string reference_file;
  std::string query_file;

  //std::string result_file = fx_param_str(module, "result_file", "result.txt");
  //std::string reference_file = fx_param_str_req(module, "reference_file");
  Matrix reference_data;
  ArrayList<index_t> neighbors;
  ArrayList<double> distances;

  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "Help options")
      ("reference_file", boost_po::value<std::string>(&reference_file), "  Reference file name")
      ("result_file", boost_po::value<std::string>(&result_file), "  Result file name")
      ("query_file", boost_po::value<std::string>(&query_file), "  Query file name")
      ("leaf_size", boost_po::value<int>()->default_value(20), "  The number of leaves")
      ("kfns", boost_po::value<int>()->default_value(1), " Value of k");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

  if( vm.count("help"))
  {
    cout  << desc << endl;
    exit(1);
  }

  if (data::Load(reference_file.c_str(), &reference_data)==SUCCESS_FAIL) {
    FATAL("Reference file %s not found", reference_file.c_str());
  }
  NOTIFY("Loaded reference data from file %s", reference_file.c_str());
 
  AllkFN allkfn; 
  //if (fx_param_exists(module, "query_file")) {
  if ( 0 != vm.count("query_file") ) {
    //std::string query_file=fx_param_str_req(module, "query_file");
   std::string query_file = vm["query_file"].as<std::string>(); 
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
  //index_t kfns=fx_param_int_req(module, "kfns");
  index_t kfns = vm["kfns"].as<index_t>();
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
