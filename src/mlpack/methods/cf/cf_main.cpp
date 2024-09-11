/**
 * @file methods/cf/cf_main.cpp
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

#undef BINDING_NAME
#define BINDING_NAME cf

#include <mlpack/core/util/mlpack_main.hpp>

#include "cf.hpp"
#include "cf_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Collaborative Filtering");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of several collaborative filtering (CF) techniques for "
    "recommender systems.  This can be used to train a new CF model, or use an"
    " existing CF model to compute recommendations.");

// Long description.
BINDING_LONG_DESC(
    "This program performs collaborative "
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
    "to be considered when generating recommendations can be specified with "
    "the " + PRINT_PARAM_STRING("neighborhood") + " parameter."
    "\n\n"
    "For performing the matrix decomposition, the following optimization "
    "algorithms can be specified via the " + PRINT_PARAM_STRING("algorithm") +
    " parameter:\n"
    "\n"
    " - 'RegSVD' -- Regularized SVD using a SGD optimizer\n"
    " - 'NMF' -- Non-negative matrix factorization with alternating least "
    "squares update rules\n"
    " - 'BatchSVD' -- SVD batch learning\n"
    " - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning\n"
    " - 'SVDCompleteIncremental' -- SVD complete incremental learning\n"
    " - 'BiasSVD' -- Bias SVD using a SGD optimizer\n"
    " - 'SVDPP' -- SVD++ using a SGD optimizer\n"
    " - 'RandSVD' -- RandomizedSVD learning\n"
    " - 'QSVD' -- QuicSVD learning\n"
    " - 'BKSVD' -- Block Krylov SVD learning\n"
    "\n\n"
    "The following neighbor search algorithms can be specified via" +
    " the " + PRINT_PARAM_STRING("neighbor_search") + " parameter:\n"
    "\n"
    " - 'cosine'  -- Cosine Search Algorithm\n"
    " - 'euclidean'  -- Euclidean Search Algorithm\n"
    " - 'pearson'  -- Pearson Search Algorithm\n"
    "\n\n"
    "The following weight interpolation algorithms can be specified via" +
    " the " + PRINT_PARAM_STRING("interpolation") + " parameter:\n"
    "\n"
    " - 'average'  -- Average Interpolation Algorithm\n"
    " - 'regression'  -- Regression Interpolation Algorithm\n"
    " - 'similarity'  -- Similarity Interpolation Algorithm\n"
    "\n\n"
    "The following ranking normalization algorithms can be specified via" +
    " the " + PRINT_PARAM_STRING("normalization") + " parameter:\n"
    "\n"
    " - 'none'  -- No Normalization\n"
    " - 'item_mean'  -- Item Mean Normalization\n"
    " - 'overall_mean'  -- Overall Mean Normalization\n"
    " - 'user_mean'  -- User Mean Normalization\n"
    " - 'z_score'  -- Z-Score Normalization\n"
    "\n"
    "A trained model may be saved to with the " +
    PRINT_PARAM_STRING("output_model") + " output parameter.");

// Example.
BINDING_EXAMPLE(
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

// See also...
BINDING_SEE_ALSO("Collaborative Filtering on Wikipedia",
        "https://en.wikipedia.org/wiki/Collaborative_filtering");
BINDING_SEE_ALSO("Matrix factorization on Wikipedia",
        "https://en.wikipedia.org/wiki/Matrix_factorization_"
        "(recommender_systems)");
BINDING_SEE_ALSO("Matrix factorization techniques for recommender systems"
        " (pdf)", "https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf"
        "&doi=cf17f85a0a7991fa01dbfb3e5878fbf71ea4bdc5");
BINDING_SEE_ALSO("CFType class documentation", "@src/mlpack/methods/cf/cf.hpp");

// Parameters for training a model.
PARAM_MATRIX_IN("training", "Input dataset to perform CF on.", "t");
PARAM_STRING_IN("algorithm", "Algorithm used for matrix factorization.", "a",
    "NMF");
PARAM_STRING_IN("normalization", "Normalization performed on the ratings.", "z",
    "none");
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
PARAM_MODEL_IN(CFModel, "input_model", "Trained CF model to load.", "m");
PARAM_MODEL_OUT(CFModel, "output_model", "Output for trained CF model.", "M");

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

//  Interpolation and Neighbor Search Algorithms
PARAM_STRING_IN("interpolation", "Algorithm used for weight interpolation.",
    "i", "average");

PARAM_STRING_IN("neighbor_search", "Algorithm used for neighbor search.",
    "S", "euclidean");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") == 0)
    RandomSeed(std::time(NULL));
  else
    RandomSeed(params.Get<int>("seed"));

  // Validate parameters.
  RequireOnlyOnePassed(params, { "training", "input_model" }, true);

  // Check that nothing stupid is happening.
  if (params.Has("query") || params.Has("all_user_recommendations"))
    RequireOnlyOnePassed(params, { "query", "all_user_recommendations" }, true);

  RequireAtLeastOnePassed(params, { "output", "output_model" }, false,
      "no output will be saved");
  if (!params.Has("query") && !params.Has("all_user_recommendations"))
    ReportIgnoredParam(params, "output", "no recommendations requested");

  RequireParamInSet<string>(params, "algorithm", { "NMF", "BatchSVD",
      "SVDIncompleteIncremental", "SVDCompleteIncremental", "RegSVD",
      "RandSVD", "BiasSVD", "SVDPP", "QSVD", "BKSVD" }, true,
      "unknown algorithm");

  ReportIgnoredParam(params, {{ "iteration_only_termination", true }},
      "min_residue");

  RequireParamValue<int>(params, "recommendations",
      [](int x) { return x > 0; }, true, "recommendations must be positive");

  // Either load from a model, or train a model.
  CFModel* cf;
  if (params.Has("training"))
  {
    // Train a model.
    // Validate Parameters.
    ReportIgnoredParam(params, {{ "iteration_only_termination", true }},
        "min_residue");
    RequireParamValue<int>(params, "rank", [](int x) { return x >= 0; }, true,
        "rank must be non-negative");
    RequireParamValue<double>(params, "min_residue",
        [](double x) { return x >= 0; }, true,
        "min_residue must be non-negative");
    RequireParamValue<int>(params, "max_iterations",
        [](int x) { return x >= 0; }, true,
        "max_iterations must be non-negative");
    RequireParamValue<int>(params, "neighborhood",
        [](int x) { return x > 0; }, true, "neighborhood must be positive");

    // Read from the input file.
    arma::mat dataset = std::move(params.Get<arma::mat>("training"));

    RequireParamValue<int>(params, "neighborhood",
        [&dataset](int x) { return x <= max(dataset.row(0)) + 1; }, true,
        "neighborbood must be less than or equal to the number of users");

    // Recommendation matrix.
    arma::Mat<size_t> recommendations;

    // Get parameters.
    const size_t rank = (size_t) params.Get<int>("rank");

    cf = new CFModel();

    // Perform decomposition to prepare for recommendations.
    Log::Info << "Performing CF matrix decomposition on dataset..." << endl;

    const string algo = params.Get<string>("algorithm");
    if (algo == "NMF")
    {
      cf->DecompositionType() = CFModel::NMF;
    }
    else if (algo == "BatchSVD")
    {
      cf->DecompositionType() = CFModel::BATCH_SVD;
    }
    else if (algo == "SVDIncompleteIncremental")
    {
      cf->DecompositionType() = CFModel::SVD_INCOMPLETE;
    }
    else if (algo == "SVDCompleteIncremental")
    {
      cf->DecompositionType() = CFModel::SVD_COMPLETE;
    }
    else if (algo == "RegSVD")
    {
      ReportIgnoredParam(params, "min_residue", "Regularized SVD terminates "
          "only when max_iterations is reached");
      cf->DecompositionType() = CFModel::REG_SVD;
    }
    else if (algo == "RandSVD")
    {
      ReportIgnoredParam(params, "min_residue", "Randomized SVD terminates "
          "only when max_iterations is reached");
      cf->DecompositionType() = CFModel::RANDOMIZED_SVD;
    }
    else if (algo == "BiasSVD")
    {
      ReportIgnoredParam(params, "min_residue", "Bias SVD terminates only "
          "when max_iterations is reached");
      cf->DecompositionType() = CFModel::BIAS_SVD;
    }
    else if (algo == "SVDPP")
    {
      ReportIgnoredParam(params, "min_residue", "SVD++ terminates only "
          "when max_iterations is reached");
      cf->DecompositionType() = CFModel::SVD_PLUS_PLUS;
    }
    else if (algo == "QSVD")
    {
      ReportIgnoredParam(params, "min_residue", "QSVD terminates only "
          "when max_iterations is reached");
      cf->DecompositionType() = CFModel::QUIC_SVD;
    }
    else if (algo == "BKSVD")
    {
      ReportIgnoredParam(params, "min_residue", "BKSVD terminates only "
          "when max_iterations is reached");
      cf->DecompositionType() = CFModel::BLOCK_KRYLOV_SVD;
    }

    // Perform the factorization and do whatever the user wanted.
    const size_t neighborhood = (size_t) params.Get<int>("neighborhood");

    // Make sure the normalization strategy is valid.
    RequireParamInSet<string>(params, "normalization", { "overall_mean",
        "item_mean", "user_mean", "z_score", "none" }, true,
        "unknown normalization type");

    const string normalizationType = params.Get<string>("normalization");
    if (normalizationType == "none")
      cf->NormalizationType() = CFModel::NO_NORMALIZATION;
    else if (normalizationType == "item_mean")
      cf->NormalizationType() = CFModel::ITEM_MEAN_NORMALIZATION;
    else if (normalizationType == "user_mean")
      cf->NormalizationType() = CFModel::USER_MEAN_NORMALIZATION;
    else if (normalizationType == "overall_mean")
      cf->NormalizationType() = CFModel::OVERALL_MEAN_NORMALIZATION;
    else if (normalizationType == "z_score")
      cf->NormalizationType() = CFModel::Z_SCORE_NORMALIZATION;

    timers.Start("cf_factorization");
    cf->Train(dataset,
              neighborhood,
              rank,
              size_t(params.Get<int>("max_iterations")),
              params.Get<double>("min_residue"),
              params.Has("iteration_only_termination"));
    timers.Stop("cf_factorization");
  }
  else
  {
    // Load from a model after validating parameters.
    RequireAtLeastOnePassed(params, { "query", "all_user_recommendations",
        "test" }, true);

    // Load an input model.
    cf = std::move(params.Get<CFModel*>("input_model"));
  }

  // Get the types of the neighbor search method and the interpolation.  (These
  // may or may not be used.)
  NeighborSearchTypes nsType;
  RequireParamInSet<string>(params, "neighbor_search", { "cosine",
      "euclidean", "pearson" }, true, "unknown neighbor search algorithm");
  if (params.Get<std::string>("neighbor_search") == "cosine")
    nsType = COSINE_SEARCH;
  else if (params.Get<std::string>("neighbor_search") == "euclidean")
    nsType = EUCLIDEAN_SEARCH;
  else // if (params.Get<std::string>("neighbor_search") == "pearson")
    nsType = PEARSON_SEARCH;

  InterpolationTypes interpolationType;
  RequireParamInSet<string>(params, "interpolation", { "average",
      "regression", "similarity" }, true, "unknown interpolation algorithm");
  if (params.Get<std::string>("interpolation") == "average")
    interpolationType = AVERAGE_INTERPOLATION;
  else if (params.Get<std::string>("interpolation") == "regression")
    interpolationType = REGRESSION_INTERPOLATION;
  else // if (params.Get<std::string>("interpolation") == "similarity")
    interpolationType = SIMILARITY_INTERPOLATION;

  if (params.Has("query") || params.Has("all_user_recommendations"))
  {
    // Get parameters for generating recommendations.
    const size_t numRecs = (size_t) params.Get<int>("recommendations");

    // Get the recommendations.
    arma::Mat<size_t> recommendations;

    // Reading users.
    if (params.Has("query"))
    {
      // User matrix.
      arma::Mat<size_t> users =
          std::move(params.Get<arma::Mat<size_t>>("query"));
      if (users.n_rows > 1)
      {
        users = users.t();
      }

      if (users.n_rows > 1)
      {
        Log::Fatal << "List of query users must be one-dimensional!"
            << std::endl;
      }

      Log::Info << "Generating recommendations for " << users.n_elem
          << " users." << endl;

      cf->GetRecommendations(nsType, interpolationType, numRecs,
          recommendations, users.row(0).t());
    }
    else
    {
      Log::Info << "Generating recommendations for all users." << endl;
      cf->GetRecommendations(nsType, interpolationType, numRecs,
          recommendations);
    }

    // Save the output.
    params.Get<arma::Mat<size_t>>("output") = recommendations;
  }

  if (params.Has("test"))
  {
    // Now, compute each test point.
    arma::mat testData = std::move(params.Get<arma::mat>("test"));

    // Assemble the combination matrix to get RMSE value.
    arma::Mat<size_t> combinations(2, testData.n_cols);
    for (size_t i = 0; i < testData.n_cols; ++i)
    {
      combinations(0, i) = size_t(testData(0, i));
      combinations(1, i) = size_t(testData(1, i));
    }

    // Now compute the RMSE.
    arma::vec predictions;
    cf->Predict(nsType, interpolationType, combinations, predictions);

    // Compute the root of the sum of the squared errors, divide by the number
    // of points to get the RMSE.  It turns out this is just the L2-norm divided
    // by the square root of the number of points, if we interpret the
    // predictions and the true values as vectors.
    const double rmse = norm(predictions - testData.row(2).t(), 2) /
        std::sqrt((double) testData.n_cols);

    Log::Info << "RMSE is " << rmse << "." << endl;
  }

  params.Get<CFModel*>("output_model") = cf;
}
