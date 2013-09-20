/**
 * @file cf_main.hpp
 * @author Mudit Raj Gupta
 *
 * Main executable to run CF.
 *
 */ 

#include <mlpack/core.hpp>
#include "cf.hpp"

using namespace mlpack;
using namespace mlpack::cf;
using namespace std;

// Document program.
PROGRAM_INFO("Collaborating Filtering", "This program performs Collaborative "
    "filtering(cf) on the given dataset. Given a list of user, item and "
    "preferences the program output is a set of recommendations for users."
    " Optionally, the users to be queried can be specified. The program also"
    " provides the flexibility to select number of recommendations for each"
    " user and also the neighbourhood. User, Item and Rating matrices can also"
    " be extracted. Variable parameters include algorithm for performing " 
    "cf, algorithm parameters and similarity measures to give recommendations");

// Parameters for program.
PARAM_STRING_REQ("input_file", "Input dataset to perform CF on.", "i");
PARAM_STRING("output_file","Recommendations.", "o", "recommendations.csv");
PARAM_STRING("algorithm", "Algorithm used for cf (als/svd).", "a","als");
PARAM_STRING("ratings_file", "File to save ratings.", "R","ratings.csv");
PARAM_STRING("user_file", "File to save the calculated User matrix to.", 
             "U","user.csv");
PARAM_STRING("item_file", "File to save the calculated Item matrix to.", 
             "I","item.csv");
PARAM_STRING("nearest_neighbour_algorithm", "Similarity Measure to be used "
             "for generating recommendations", "s","knn");
PARAM_STRING("query_file", "List of user for which recommendations are to "
             "be generated (If unspecified then all)", "q","query.csv");
PARAM_INT("number_of_Recommendations", "Number of Recommendations for each "
          "user in query", "r",5);
PARAM_INT("neighbourhood", "Size of the neighbourhood for all "
          "user in query", "n",5);

int main(int argc, char** argv)
{
  //Parse Command Line
  CLI::ParseCommandLine(argc, argv);
 
  //Read from the input file
  string inputFile = CLI::GetParam<string>("input_file");
  arma::mat dataset;
  data::Load(inputFile.c_str(), dataset);
    
  //Recommendation matrix.
  arma::Mat<size_t> recommendations; 
   
  //User Matrix
  arma::Col<size_t> users;  

  //Reading Users
  string userf = CLI::GetParam<string>("query_file");
  data::Load(userf.c_str(),users);

  //Calculating Recommendations
  CF c(dataset);

  Log::Info << "Performing CF on dataset..." << endl;
  c.GetRecommendations(recommendations);
  string outputFile = CLI::GetParam<string>("output_file");
  data::Save(outputFile, recommendations);
}
