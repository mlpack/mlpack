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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/regularized_svd/regularized_svd.hpp>
#include <mlpack/methods/amf/termination_policies/max_iteration_termination.hpp>
#include "cf.hpp"

using namespace mlpack;
using namespace mlpack::cf;
using namespace mlpack::amf;
using namespace mlpack::svd;
using namespace mlpack::util;
using namespace std;

// Document program.
PROGRAM_INFO("Collaborative Filtering", "This program performs collaborative "
    "filtering (CF) on the given dataset. Given a list of user, item and "
    "preferences (the " + PRINT_PARAM_STRING("training") + " parameter), "
    "the program will perform a matrix decomposition and then can perform a "
    "series of actions related to collaborative filtering.  Alternately, the "
    "program can load an existing saved CF model with the " +
    PRINT_PARAM_STRING("input_model") + " parameter and then use that model "
    "to provide recommendations or predict values."
    "\n\n"
    "The input matrix should be a 3-dimensional matrix of ratings, where the "
    "first dimension is the user, the second dimension is the item, and the "
    "third dimension is that user's rating of that item.  Both the users and "
    "items should be numeric indices, not names. The indices are assumed to "
    "start from 0."
    "\n\n"
    "A set of query users for which recommendations can be generated may be "
    "specified with the " + PRINT_PARAM_STRING("query") + " parameter; "
    "alternately, recommendations may be generated for every user in the "
    "dataset by specifying the " +
    PRINT_PARAM_STRING("all_user_recommendations") + " parameter.  In "
    "addition, the number of recommendations per user to generate can be "
    "specified with the " + PRINT_PARAM_STRING("recommendations") + " "
    "parameter, and the number of similar users (the size of the neighborhood) "
    " to be considered when generating recommendations can be specified with "
    "the " + PRINT_PARAM_STRING("neighborhood") + " parameter."
    "\n\n"
    "For performing the matrix decomposition, the following optimization "
    "algorithms can be specified via the " + PRINT_PARAM_STRING("algorithm") +
    " parameter: "
    "\n"
    " - 'RegSVD' -- Regularized SVD using a SGD optimizer\n"
    " - 'NMF' -- Non-negative matrix factorization with alternating least "
    "squares update rules\n"
    " - 'BatchSVD' -- SVD batch learning\n"
    " - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning\n"
    " - 'SVDCompleteIncremental' -- SVD complete incremental learning\n"
    "\n"
    "A trained model may be saved to with the " +
    PRINT_PARAM_STRING("output_model") + " output parameter."
    "\n\n"
    "To train a CF model on a dataset " + PRINT_DATASET("training_set") + " "
    "using NMF for decomposition and saving the trained model to " +
    PRINT_MODEL("model") + ", one could call: "
    "\n\n" +
    PRINT_CALL("cf", "training", "training_set", "algorithm", "NMF",
        "output_model", "model") +
    "\n\n"
    "Then, to use this model to generate recommendations for the list of users "
    "in the query set " + PRINT_DATASET("users") + ", storing 5 "
    "recommendations in " + PRINT_DATASET("recommendations") + ", one could "
    "call "
    "\n\n" +
    PRINT_CALL("cf", "input_model", "model", "query", "users",
        "recommendations", 5, "output", "recommendations"));

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
PARAM_INT_IN("max_iterations", "Maximum number of iterations. If set to zero, "
    "there is no limit on the number of iterations.", "N", 1000);
PARAM_FLAG("iteration_only_termination", "Terminate only when the maximum "
    "number of iterations is reached.", "I");
PARAM_DOUBLE_IN("min_residue", "Residue required to terminate the factorization"
    " (lower values generally mean better fits).", "r", 1e-5);

// Load/save a model.
PARAM_MODEL_IN(CF, "input_model", "Trained CF model to load.", "m");
PARAM_MODEL_OUT(CF, "output_model", "Output for trained CF model.", "M");

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

void ComputeRecommendations(CF* cf,
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
    cf->GetRecommendations(numRecs, recommendations, users.row(0).t());
  }
  else
  {
    Log::Info << "Generating recommendations for all users." << endl;
    cf->GetRecommendations(numRecs, recommendations);
  }
}

void ComputeRMSE(CF* cf)
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
  cf->Predict(combinations, predictions);

  // Compute the root of the sum of the squared errors, divide by the number of
  // points to get the RMSE.  It turns out this is just the L2-norm divided by
  // the square root of the number of points, if we interpret the predictions
  // and the true values as vectors.
  const double rmse = arma::norm(predictions - testData.row(2).t(), 2) /
      std::sqrt((double) testData.n_cols);

  Log::Info << "RMSE is " << rmse << "." << endl;
}

void PerformAction(CF* c)
{
  if (CLI::HasParam("query") || CLI::HasParam("all_user_recommendations"))
  {
    // Get parameters for generating recommendations.
    const size_t numRecs = (size_t) CLI::GetParam<int>("recommendations");

    // Get the recommendations.
    arma::Mat<size_t> recommendations;
    ComputeRecommendations(c, numRecs, recommendations);

    // Save the output.
    CLI::GetParam<arma::Mat<size_t>>("output") = recommendations;
  }

  if (CLI::HasParam("test"))
    ComputeRMSE(c);

  CLI::GetParam<CF*>("output_model") = c;
}

template<typename Factorizer>
void PerformAction(Factorizer&& factorizer,
                   arma::mat& dataset,
                   const size_t rank)
{
  // Parameters for generating the CF object.
  const size_t neighborhood = (size_t) CLI::GetParam<int>("neighborhood");
  CF* c = new CF(dataset, factorizer, neighborhood, rank);

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
    else if (algorithm == "BatchSVD")
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
      ReportIgnoredParam("min_residue", "Regularized SVD terminates only "
          "when max_iterations is reached");
      PerformAction(RegularizedSVD<>(maxIterations), dataset, rank);
    }
  }
  else
  {
    // Use default termination (SimpleResidueTermination), but set the maximum
    // number of iterations.
    const double minResidue = CLI::GetParam<double>("min_residue");
    SimpleResidueTermination srt(minResidue, maxIterations);
    if (algorithm == "NMF")
    {
      PerformAction(NMFALSFactorizer(srt), dataset, rank);
    }
    else if (algorithm == "BatchSVD")
    {
      PerformAction(SVDBatchFactorizer<>(srt), dataset, rank);
    }
    else if (algorithm == "SVDIncompleteIncremental")
    {
      PerformAction(SVDIncompleteIncrementalFactorizer<arma::sp_mat>(srt),
          dataset, rank);
    }
    else if (algorithm == "SVDCompleteIncremental")
    {
      PerformAction(SVDCompleteIncrementalFactorizer<arma::sp_mat>(srt),
          dataset, rank);
    }
    else if (algorithm == "RegSVD")
    {
      ReportIgnoredParam("min_residue", "Regularized SVD terminates only "
          "when max_iterations is reached");
      PerformAction(RegularizedSVD<>(maxIterations), dataset, rank);
    }
  }
}

static void mlpackMain()
{
  if (CLI::GetParam<int>("seed") == 0)
    math::RandomSeed(std::time(NULL));
  else
    math::RandomSeed(CLI::GetParam<int>("seed"));

  // Validate parameters.
  RequireOnlyOnePassed({ "training", "input_model" }, true);

  // Check that nothing stupid is happening.
  if (CLI::HasParam("query") || CLI::HasParam("all_user_recommendations"))
    RequireOnlyOnePassed({ "query", "all_user_recommendations" }, true);

  RequireAtLeastOnePassed({ "output", "output_model" }, false,
      "no output will be saved");
  if (!CLI::HasParam("query") && !CLI::HasParam("all_user_recommendations"))
    ReportIgnoredParam("output", "no recommendations requested");

  RequireParamInSet<string>("algorithm", { "NMF", "BatchSVD",
      "SVDIncompleteIncremental", "SVDCompleteIncremental", "RegSVD" }, true,
      "unknown algorithm");

  ReportIgnoredParam({{ "iteration_only_termination", true }}, "min_residue");

  RequireParamValue<int>("recommendations", [](int x) { return x > 0; }, true,
        "recommendations must be positive");

  // Either load from a model, or train a model.
  if (CLI::HasParam("training"))
  {
    // Train a model.
    // Validate Parameters.
    ReportIgnoredParam({{ "iteration_only_termination", true }}, "min_residue");
    RequireParamValue<int>("rank", [](int x) { return x >= 0; }, true,
        "rank must be non-negative");
    RequireParamValue<double>("min_residue", [](double x) { return x >= 0; },
        true, "min_residue must be non-negative");
    RequireParamValue<int>("max_iterations", [](int x) { return x >= 0; }, true,
        "max_iterations must be non-negative");
    RequireParamValue<int>("neighborhood", [](int x) { return x > 0; }, true,
        "neighborhood must be positive");

    // Read from the input file.
    arma::mat dataset = std::move(CLI::GetParam<arma::mat>("training"));

    RequireParamValue<int>("neighborhood",
        [&dataset](int x) { return x <= max(dataset.row(0)) + 1; }, true,
        "neighborbood must be less than or equal to the number of users");

    // Recommendation matrix.
    arma::Mat<size_t> recommendations;

    // Get parameters.
    const size_t rank = (size_t) CLI::GetParam<int>("rank");

    // Perform decomposition to prepare for recommendations.
    Log::Info << "Performing CF matrix decomposition on dataset..." << endl;

    const string algo = CLI::GetParam<string>("algorithm");

    // Perform the factorization and do whatever the user wanted.
    AssembleFactorizerType(algo, dataset,
        CLI::HasParam("iteration_only_termination"), rank);
  }
  else
  {
    // Load from a model after validating parameters.
    RequireAtLeastOnePassed({ "query", "all_user_recommendations",
        "test" }, true);

    // Load an input model.
    CF* c = std::move(CLI::GetParam<CF*>("input_model"));

    PerformAction(c);
  }
}
