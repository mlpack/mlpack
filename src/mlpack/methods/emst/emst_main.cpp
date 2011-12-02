/**
 * @file emst.cc
 *
 * Calls the DualTreeBoruvka algorithm from dtb.h
 * Can optionally call Naive Boruvka's method
 *
 * For algorithm details, see:
 * March, W.B., Ram, P., and Gray, A.G.
 * Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, Applications.
 * In KDD, 2010.
 *
 * @author Bill March (march@gatech.edu)
 */

#include "dtb.hpp"

#include <mlpack/core.hpp>

PARAM_STRING_REQ("input_file", "Data input file.", "emst");
PARAM_STRING("output_file", "Data output file.  Stored as an edge list.", "emst", "emst_output.csv");
PARAM_FLAG("do_naive", "Compute the MST using .", "naive");
PARAM_STRING("output_file", "Naive data output file.", "naive",
    "naive_output.csv");
PARAM_INT("leaf_size", "Leaf size in the kd-tree.  Singleton leaves give the empirically best performance at the cost of greater memory requirements.", "emst", 1);
PARAM_DOUBLE("total_squared_length", "Squared length of the computed tree.", "dtb", 0.0);

using namespace mlpack;
using namespace mlpack::emst;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  ///////////////// READ IN DATA //////////////////////////////////
  std::string data_file_name = CLI::GetParam<std::string>("emst/input_file");

  Log::Info << "Reading in data.\n";

  arma::mat data_points;
  data::Load(data_file_name.c_str(), data_points, true);

  // Do naive
  if (CLI::GetParam<bool>("naive/do_naive"))
  {
    Log::Info << "Running naive algorithm.\n";

    DualTreeBoruvka naive;

    naive.Init(data_points, true);

    arma::mat naive_results;
    naive.ComputeMST(naive_results);

    std::string naive_output_filename =
    CLI::GetParam<std::string>("naive/output_file");

    data::Save(naive_output_filename.c_str(), naive_results, true);
  }
  else
  {
    Log::Info << "Data read, building tree.\n";

    /////////////// Initialize DTB //////////////////////
    size_t leafSize = CLI::GetParam<int>("emst/leaf_size");
    DualTreeBoruvka dtb;
    dtb.Init(data_points, false, leafSize);

    Log::Info << "Tree built, running algorithm.\n\n";

    ////////////// Run DTB /////////////////////
    arma::mat results;

    dtb.ComputeMST(results);


    //////////////// Output the Results ////////////////

    std::string output_filename =
        CLI::GetParam<std::string>("emst/output_file");

    data::Save(output_filename.c_str(), results, true);
  }

  return 0;
}
