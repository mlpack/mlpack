/**
 * @file cf_main.hpp
 * @author Mudit Raj Gupta
 *
 * Main executable to run CF.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
    "preferences (--training_file) the program will perform a matrix "
    "decomposition and then can perform a series of actions related to "
    "collaborative filtering.  Alternately, the program can load an existing "
    "saved CF model with the --input_model_file (-m) option and then use that "
    "model to provide recommendations or predict values."
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
    "'SVDCompleteIncremental' -- SVD complete incremental learning\n"
    "\n"
    "A trained model may be saved to a file with the --output_model_file (-M) "
    "parameter.");

// Parameters for training a model.
PARAM_MATRIX_IN("training", "Input dataset to perform CF on.", "t");
PARAM_STRING_IN("algorithm", "Algorithm used for matrix factorization.", "a",
    "NMF");
PARAM_INT_IN("neighborhood", "Size of the neighborhood of similar users to "
    "consider for each query user.", "n", 5);
PARAM_INT_IN("rank", "Rank of decomposed matrices (if 0, a heuristic is used to"
    " estimate the rank).", "R", 0);
PARAM_MATRIX_IN("test", "Test set to calculate RMSE on.", "T");

// Offer the user the option to set the maximum number of iterations, and
// terminate only based on the number of iterations.
PARAM_INT_IN("max_iterations", "Maximum number of iterations.", "N", 1000);
PARAM_FLAG("iteration_only_termination", "Terminate only when the maximum "
    "number of iterations is reached.", "I");
PARAM_DOUBLE_IN("min_residue", "Residue required to terminate the factorization"
    " (lower values generally mean better fits).", "r", 1e-5);

// Load/save a model.
PARAM_STRING_IN("input_model_file", "File to load trained CF model from.", "m",
    "");
PARAM_STRING_OUT("output_model_file", "File to save trained CF model to.", "M");

// Query settings.
PARAM_UMATRIX_IN("query", "List of query users for which recommendations should"
    " be generated.", "q");
PARAM_FLAG("all_user_recommendations", "Generate recommendations for all "
    "users.", "A");
PARAM_UMATRIX_OUT("output", "Matrix that will store output recommendations.",
    "o");
PARAM_INT_IN("recommendations", "Number of recommendations to generate for each"
    " query user.", "c", 5);

PARAM_INT_IN("seed", "Set the random seed (0 uses std::time(NULL)).", "s", 0);

void ComputeRecommendations(CF& cf,
                            const size_t numRecs,
                            arma::Mat<size_t>& recommendations)
{
  // Reading users.
  if (CLI::HasParam("query"))
  {
    // User matrix.
    arma::Mat<size_t> users =
        std::move(CLI::GetParam<arma::Mat<size_t>>("query"));
    if (users.n_rows > 1)
      users = users.t();
    if (users.n_rows > 1)
      Log::Fatal << "List of query users must be one-dimensional!" << std::endl;

    Log::Info << "Generating recommendations for " << users.n_elem << " users."
        << endl;
    cf.GetRecommendations(numRecs, recommendations, users.row(0).t());
  }
  else
  {
    Log::Info << "Generating recommendations for all users." << endl;
    cf.GetRecommendations(numRecs, recommendations);
  }
}

void ComputeRMSE(CF& cf)
{
  // Now, compute each test point.
  arma::mat testData = std::move(CLI::GetParam<arma::mat>("test"));

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

void PerformAction(CF& c)
{
  if (CLI::HasParam("query") || CLI::HasParam("all_user_recommendations"))
  {
    // Get parameters for generating recommendations.
    const size_t numRecs = (size_t) CLI::GetParam<int>("recommendations");

    // Get the recommendations.
    arma::Mat<size_t> recommendations;
    ComputeRecommendations(c, numRecs, recommendations);

    // Save the output.
    if (CLI::HasParam("output"))
      CLI::GetParam<arma::Mat<size_t>>("output") = recommendations;
  }

  if (CLI::HasParam("test"))
    ComputeRMSE(c);

  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "cf_model", c);
}

template<typename Factorizer>
void PerformAction(Factorizer&& factorizer,
                   arma::mat& dataset,
                   const size_t rank)
{
  // Parameters for generating the CF object.
  const size_t neighborhood = (size_t) CLI::GetParam<int>("neighborhood");
  CF c(dataset, factorizer, neighborhood, rank);

  PerformAction(c);
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

  if (CLI::GetParam<int>("seed") == 0)
    math::RandomSeed(std::time(NULL));
  else
    math::RandomSeed(CLI::GetParam<int>("seed"));

  // Validate parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Only one of --training_file (-t) or --input_model_file (-m) "
        << "may be specified!" << endl;

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Neither --training_file (-t) nor --input_model_file (-m) are"
        << " specified!" << endl;

  // Check that nothing stupid is happening.
  if (CLI::HasParam("query") && CLI::HasParam("all_user_recommendations"))
    Log::Fatal << "Both --query_file and --all_user_recommendations are given, "
        << "but only one is allowed!" << endl;

  if (!CLI::HasParam("output") && !CLI::HasParam("output_model_file"))
    Log::Warn << "Neither --output_file nor --output_model_file are specified; "
        << "no output will be saved." << endl;

  if (CLI::HasParam("output") && (!CLI::HasParam("query") ||
      CLI::HasParam("all_user_recommendations")))
    Log::Warn << "--output_file is ignored because neither --query_file nor "
        << "--all_user_recommendations are specified." << endl;

  // Either load from a model, or train a model.
  if (CLI::HasParam("training"))
  {
    // Read from the input file.
    arma::mat dataset = std::move(CLI::GetParam<arma::mat>("training"));

    // Recommendation matrix.
    arma::Mat<size_t> recommendations;

    // Get parameters.
    const size_t rank = (size_t) CLI::GetParam<int>("rank");

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
      Log::Warn << "--min_residue ignored, because --iteration_only_termination"
          << " is specified." << endl;

    // Perform the factorization and do whatever the user wanted.
    AssembleFactorizerType(algo, dataset,
        CLI::HasParam("iteration_only_termination"), rank);
  }
  else
  {
    // Load an input model.
    CF c;
    data::Load(CLI::GetParam<string>("input_model_file"), "cf_model", c, true);

    PerformAction(c);
  }
}
