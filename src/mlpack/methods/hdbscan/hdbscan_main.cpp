/**
 * @file hdbscan_main.cpp
 * @author Sudhanshu Ranjan
 *
 * Implementation of program to run HDBSCAN.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "hdbscan.hpp"

using namespace mlpack;
using namespace mlpack::hdbscan;
using namespace std;

PROGRAM_INFO("HDBSCAN algorithm", "This program helps in clustering "
    "and is useful when the clusters have variable density "
    "\n\n"
    "The output is saved in a column matrix");

// PARAM_STRING_OUT("output_file", 
//                  "Output file. Stores cluster label of each point.",
//                  "o");
// PARAM_STRING_IN("input_file", 
//                 "Input dataset to cluster", 
//                 "i",
//                 "noFileSelected");
// PARAM_FLAG("single_cluster", "Allow single cluster.", "s");
// PARAM_INT_IN("minimumClusterSize", "This is the minimum number of points"
//     " present in the cluster ", "m", 10);

int main(int argc, char* argv[])
{
	// CLI::ParseCommandLine(argc, argv);

 //  // Get minimum cluster size 
 //  // and whether to allow single cluster
 //  size_t minimumClusterSize = (CLI::GetParam<int>("minimumClusterSize"));
 //  bool allowSingleCluster = CLI::GetParam<bool>("single_cluster");

 //  if (CLI::GetParam<string>("input_file")=="noFileSelected")
 //    Log::Fatal<<"--input_file is not speified !"<<endl;
    
 //  // Warn if output file is not specified
 //  if (!CLI::HasParam("output_file"))
 //  	Log::Warn << "--output_file is not specified, so no output will be saved!"
	//           << endl;
    
 //  // By default single cluster is disabled
 //  if (!CLI::HasParam("single_cluster"))
 //  {
 //  	Log::Warn << "--single_cluster is not specified. Default value is false."
	// 	  <<"Set this to true only if you think it fits your dataset!"<< endl;
 //    	allowSingleCluster = false;
 //  }
    
 //  // Default value of minimum cluster size is 10
 //  if (!CLI::HasParam("minimumClusterSize"))
 //  {
	// Log::Warn << "--minimumClusterSize is not specified. Default value is 10!"
 //    		  << endl;
 //    	minimumClusterSize = 10;
 //  }

 //  arma::mat dataPoints;
 //  data::Load(CLI::GetParam<string>("input_file"), dataPoints, false);

 //  //check that minimum cluster size is positive
 //  if (CLI::GetParam<int>("minimumClusterSize") <= 0)
 //  {
 //    Log::Fatal << "Invalid cluster size (" 
 //               << CLI::GetParam<int>("minimumClusterSize")
 //               << ")!  Must be greater than or equal to 1." << std::endl;
 //  }

 //  HDBSCAN<> hdbscan(minimumClusterSize, allowSingleCluster);
 //  arma::Row<size_t> result;
 //  hdbscan.Cluster(dataPoints, result);

 //  if (CLI::HasParam("output_file"))
 //    data::Save(CLI::GetParam<string>("output_file"), result, false);
}
