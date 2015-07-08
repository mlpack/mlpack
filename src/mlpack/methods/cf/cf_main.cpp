/**
 * @file cf_main.hpp
 * @author Mudit Raj Gupta
 *
 * Main executable to run CF.
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/regularized_svd/regularized_svd.hpp>
#include <mlpack/methods/amf/termination_policies/max_iteration_termination.hpp>
#include "cf.hpp"

using namespace mlpack;
using namespace mlpack::cf;
using namespace mlpack::amf;
using namespace mlpack::svd;
using namespace std;

// Document program.
PROGRAM_INFO("Collaborating Filtering", "This program performs collaborative "
    "filtering (CF) on the given dataset. Given a list of user, item and "
    "preferences (--input_file) the program will perform a matrix decomposition"
    " and then can perform a series of actions related to collaborative "
    "filtering."
    "\n\n"
    "The input file should contain a 3-column matrix of ratings, where the "
    "first column is the user, the second column is the item, and the third "
    "column is that user's rating of that item.  Both the users and items "
    "should be numeric indices, not names. The indices are assumed to start "
    "from 0."
    "\n\n"
    "A set of query users for which recommendations can be generated may be "
    "specified with the --query_file (-q) option; alternately, recommendations "
    "may be generated for every user in the dataset by specifying the "
    "--all_user_recommendations (-A) option.  In addition, the number of "
    "recommendations per user to generate can be specified with the "
    "--recommendations (-r) parameter, and the number of similar users (the "
    "size of the neighborhood) to be considered when generating recommendations"
    " can be specified with the --neighborhood (-n) option."
    "\n\n"
    "For performing the matrix decomposition, the following optimization "
    "algorithms can be specified via the --algorithm (-a) parameter: "
    "\n"
    "'RegSVD' -- Regularized SVD using a SGD optimizer\n"
    "'NMF' -- Non-negative matrix factorization with alternating least squares "
    "update rules\n"
    "'BatchSVD' -- SVD batch learning\n"
    "'SVDIncompleteIncremental' -- SVD incomplete incremental learning\n"
    "'SVDCompleteIncremental' -- SVD complete incremental learning\n");

// Parameters for program.
PARAM_STRING_REQ("input_file", "Input dataset to perform CF on.", "i");
PARAM_STRING("query_file", "List of users for which recommendations are to "
    "be generated.", "q", "");
PARAM_FLAG("all_user_recommendations", "Generate recommendations for all "
    "users.", "A");

PARAM_STRING("output_file","File to save output recommendations to.", "o",
    "recommendations.csv");

PARAM_STRING("algorithm", "Algorithm used for matrix factorization.", "a",
    "NMF");

PARAM_INT("recommendations", "Number of recommendations to generate for each "
    "query user.", "r", 5);
PARAM_INT("neighborhood", "Size of the neighborhood of similar users to "
    "consider for each query user.", "n", 5);

PARAM_INT("rank", "Rank of decomposed matrices (if 0, a heuristic is used to "
    "estimate the rank).", "R", 0);

PARAM_STRING("test_file", "Test set to calculate RMSE on.", "t", "");

// Offer the user the option to set the maximum number of iterations, and
// terminate only based on the number of iterations.
PARAM_INT("max_iterations", "Maximum number of iterations.", "m", 1000);
PARAM_FLAG("iteration_only_termination", "Terminate only when the maximum "
    "number of iterations is reached.", "I");
PARAM_DOUBLE("min_residue", "Residue required to terminate the factorization "
    "(lower values generally mean better fits).", "r", 1e-5);

template<typename Factorizer>
void ComputeRecommendations(CF<Factorizer>& cf,
                            const size_t numRecs,
                            arma::Mat<size_t>& recommendations)
{
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
    cf.GetRecommendations(numRecs, recommendations, users);
  }
  else
  {
    Log::Info << "Generating recommendations for all users." << endl;
    cf.GetRecommendations(numRecs, recommendations);
  }
}

template<typename Factorizer>
void ComputeRMSE(CF<Factorizer>& cf)
{
  // Now, compute each test point.
  const string testFile = CLI::GetParam<string>("test_file");
  arma::mat testData;
  data::Load(testFile, testData, true);

  // Assemble the combination matrix to get RMSE value.
  arma::Mat<size_t> combinations(2, testData.n_cols);
  for (size_t i = 0; i < testData.n_cols; ++i)
  {
    combinations(0, i) = size_t(testData(0, i));
    combinations(1, i) = size_t(testData(1, i));
  }

  // Now compute the RMSE.
  arma::vec predictions;
  cf.Predict(combinations, predictions);

  // Compute the root of the sum of the squared errors, divide by the number of
  // points to get the RMSE.  It turns out this is just the L2-norm divided by
  // the square root of the number of points, if we interpret the predictions
  // and the true values as vectors.
  const double rmse = arma::norm(predictions - testData.row(2).t(), 2) /
      std::sqrt((double) testData.n_cols);

  Log::Info << "RMSE is " << rmse << "." << endl;
}

template<typename Factorizer>
void PerformAction(Factorizer&& factorizer,
                   arma::mat& dataset,
                   const size_t rank)
{
  // Parameters for generating the CF object.
  const size_t neighborhood = (size_t) CLI::GetParam<int>("neighborhood");
  CF<Factorizer> c(dataset, factorizer, neighborhood, rank);

  if (CLI::HasParam("query_file") || CLI::HasParam("all_user_recommendations"))
  {
    // Get parameters for generating recommendations.
    const size_t numRecs = (size_t) CLI::GetParam<int>("recommendations");

    // Get the recommendations.
    arma::Mat<size_t> recommendations;
    ComputeRecommendations(c, numRecs, recommendations);

    // Save the output.
    const string outputFile = CLI::GetParam<string>("output_file");
    data::Save(outputFile, recommendations);
  }

  if (CLI::HasParam("test_file"))
  {
    ComputeRMSE(c);
  }
}

void AssembleFactorizerType(const std::string& algorithm,
                            arma::mat& dataset,
                            const bool maxIterationTermination,
                            const size_t rank)
{
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  if (maxIterationTermination)
  {
    // Force termination when maximum number of iterations reached.
    MaxIterationTermination mit(maxIterations);
    if (algorithm == "NMF")
    {
      typedef AMF<MaxIterationTermination, RandomInitialization, NMFALSUpdate>
          FactorizerType;
      PerformAction(FactorizerType(mit), dataset, rank);
    }
    else if (algorithm == "SVDBatch")
    {
      typedef AMF<MaxIterationTermination, RandomInitialization,
          SVDBatchLearning> FactorizerType;
      PerformAction(FactorizerType(mit), dataset, rank);
    }
    else if (algorithm == "SVDIncompleteIncremental")
    {
      typedef AMF<MaxIterationTermination, RandomInitialization,
          SVDIncompleteIncrementalLearning> FactorizerType;
      PerformAction(FactorizerType(mit), dataset, rank);
    }
    else if (algorithm == "SVDCompleteIncremental")
    {
      typedef AMF<MaxIterationTermination, RandomInitialization,
          SVDCompleteIncrementalLearning<arma::sp_mat>> FactorizerType;
      PerformAction(FactorizerType(mit), dataset, rank);
    }
    else if (algorithm == "RegSVD")
    {
      Log::Fatal << "--iteration_only_termination not supported with 'RegSVD' "
          << "algorithm!" << endl;
    }
  }
  else
  {
    // Use default termination (SimpleResidueTermination), but set the maximum
    // number of iterations.
    const double minResidue = CLI::GetParam<double>("min_residue");
    SimpleResidueTermination srt(minResidue, maxIterations);
    if (algorithm == "NMF")
      PerformAction(NMFALSFactorizer(srt), dataset, rank);
    else if (algorithm == "SVDBatch")
      PerformAction(SVDBatchFactorizer(srt), dataset, rank);
    else if (algorithm == "SVDIncompleteIncremental")
      PerformAction(SparseSVDIncompleteIncrementalFactorizer(srt), dataset,
          rank);
    else if (algorithm == "SVDCompleteIncremental")
      PerformAction(SparseSVDCompleteIncrementalFactorizer(srt), dataset, rank);
    else if (algorithm == "RegSVD")
      PerformAction(RegularizedSVD<>(maxIterations), dataset, rank);
  }
}

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

  // Get parameters.
  const size_t rank = (size_t) CLI::GetParam<int>("rank");

  // Check that nothing stupid is happening.
  if (CLI::HasParam("query_file") && CLI::HasParam("all_user_recommendations"))
    Log::Fatal << "Both --query_file and --all_user_recommendations are given, "
        << "but only one is allowed!" << endl;

  // Perform decomposition to prepare for recommendations.
  Log::Info << "Performing CF matrix decomposition on dataset..." << endl;

  const string algo = CLI::GetParam<string>("algorithm");

  // Issue an error if an invalid factorizer is used.
  if (algo != "NMF" &&
      algo != "SVDBatch" &&
      algo != "SVDIncompleteIncremental" &&
      algo != "SVDCompleteIncremental" &&
      algo != "RegSVD")
    Log::Fatal << "Invalid decomposition algorithm.  Choices are 'NMF', "
        << "'SVDBatch', 'SVDIncompleteIncremental', 'SVDCompleteIncremental',"
        << " and 'RegSVD'." << endl;

  // Issue a warning if the user provided a minimum residue but it will be
  // ignored.
  if (CLI::HasParam("min_residue") &&
      CLI::HasParam("iteration_only_termination"))
    Log::Warn << "--min_residue ignored, because --iteration_only_termination "
        << "is specified." << endl;

  // Perform the factorization and do whatever the user wanted.
  AssembleFactorizerType(algo, dataset,
      CLI::HasParam("iteration_only_termination"), rank);
}
