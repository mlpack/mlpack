/**
 * @file mds_main.cpp
 * @author Dhawal Arora
 * This will form the executable for Multi-Dimensional Scaling
 * 
 */
#include <mlpack/core.hpp>
#include "mds.hpp"
using namespace std;
using namespace mlpack;
using namespace mlpack::manifold;
using namespace arma;

// Document program.
PROGRAM_INFO("Multi Dimensional Scaling", "Multidimensional Scaling is a classical approach"
 	     "that maps the original high dimensional space to a lower dimensional space,"
             "but does so in an attempt to preserve pairwise distances"
             "between all the data points in both the dimensions.");

// Parameters for program.
PARAM_STRING_REQ("input", "Input dataset to perform MDS on.", "i");
PARAM_STRING_REQ("output", "File to save modified dataset to.", "o");
PARAM_INT("dimensionality", "Desired dimensionality of output dataset." , "d", 2);


int main(int argc,char **argv){
    
    // Parse commandline.
    CLI::ParseCommandLine(argc, argv);
	
    // Load input dataset.
    string inputFile = CLI::GetParam<string>("input");
    arma::mat dataset;
  	
    Timer::Start("Load dataset");
    data::Load(inputFile, dataset);
    dataset=trans(dataset);
    Timer::Stop("Load dataset");

    size_t dimensions = CLI::GetParam<int>("dimensionality");

    if (dimensions >= dataset.n_rows){
      	Log::Warn << "For cases of dimensionality greater than the number of observations,"
      	"the result for dimensionality same as number of observations will be computed" << std::endl;
    }
  	
    //Perform MDS
    Timer::Start("Perform MDS");
    MDS m(dataset,dimensions,"euclidean","metric");
    dataset=m.transformedData();
    Timer::Stop("Perform MDS");

    Timer::Start("Save output");
    // Now save the results.
    string outputFile = CLI::GetParam<string>("output");
    dataset=trans(dataset);
    data::Save(outputFile, dataset);
    Timer::Stop("Save output");


    return 0;
}
