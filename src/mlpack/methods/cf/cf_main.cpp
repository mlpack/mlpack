/**
 * @file cf_main.hpp
 * @author Mudit Raj Gupta
 *
 * Main executable to run CF.
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
PARAM_STRING("query_file", "List of users for which recommendations are to "
    "be generated (if unspecified, then recommendations are generated for all "
    "users).", "q", "");

PARAM_STRING("output_file","File to save output recommendations to.", "o",
    "recommendations.csv");

// These features are not yet available in the CF code.
//PARAM_STRING("algorithm", "Algorithm used for CF ('als' or 'svd').", "a",
//    "als");
//PARAM_STRING("nearest_neighbor_algorithm", "Similarity search procedure to "
//    "be used for generating recommendations.", "s", "knn");

PARAM_INT("number_of_Recommendations", "Number of Recommendations for each "
          "user in query", "r",5);
PARAM_INT("neighbourhood", "Size of the neighbourhood for all "
          "user in query", "n",5);

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Read from the input file.
  const string inputFile = CLI::GetParam<string>("input_file");
  arma::mat dataset;
  data::Load(inputFile, dataset, true);

  // Recommendation matrix.
  arma::Mat<size_t> recommendations;

  // Perform decomposition to prepare for recommendations.
  Log::Info << "Performing CF matrix decomposition on dataset..." << endl;
  CF c(dataset);

  // Reading users.
  const string queryFile = CLI::GetParam<string>("query_file");
  if (queryFile != "")
  {
    // User matrix.
    arma::Mat<size_t> userTmp;
    arma::Col<size_t> users;
    data::Load(queryFile, userTmp, true, false /* Don't transpose. */);
    users = userTmp.col(0);

    Log::Info << "Generating recommendations for " << users.n_elem << " users "
        << "in '" << queryFile << "'." << endl;
    c.GetRecommendations(recommendations, users);
  }
  else
  {
    Log::Info << "Generating recommendations for all users." << endl;
    c.GetRecommendations(recommendations);
  }

  const string outputFile = CLI::GetParam<string>("output_file");
  data::Save(outputFile, recommendations);
}
