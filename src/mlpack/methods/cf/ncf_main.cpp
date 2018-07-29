/**
 * @file ncf_main.hpp
 * @author Haritha Nair
 *
 * Main executable to run NCF.
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

#include "ncf.hpp"

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/core/optimizers/ada_grad/ada_grad.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

using namespace mlpack;
using namespace mlpack::cf;
using namespace mlpack::util;
using namespace mlpack::optimization;
using namespace std;

// Document program.
PROGRAM_INFO("Neural collaborative Filtering", "This program performs "
    "collaborative filtering (CF) on the given dataset. Given a list of user, "
    "item and ratings (the " + PRINT_PARAM_STRING("training") + " parameter), "
    "and the kind of network to be used, the program will train and use the "
    " network and enable the model to predict further ratings. Alternately, "
    "the program can load an existing saved NCF model with the " +
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
    "parameter."
    "\n\n"
    "For creating the network the following algorithms can be used using the "
    + PRINT_PARAM_STRING("algorithm") + "parameter: "
    "\n"
    " - 'GMF' -- General matrix factorization\n"
    " - 'MLP' -- Multi layer perceptron\n"
    " - 'NeuMF' -- Neural Matrix Factorization\n"
    "\n"
    "The optimizer to be used to optimize the model can be specified using the "
    + PRINT_PARAM_STRING("optimizer") + "parameter: "
    "\n"
    " - 'adagrad'\n"
    " - 'rmsprop'\n"
    " - 'adam'\n"
    " - 'SGD'\n"
    "\n"
    "For training, the embedding to be created for each user and item data can "
    "have user specified size and can be set using the " +
    PRINT_PARAM_STRING("embedsize") + "parameter and the number of negative or "
    "unrated instances to be trained upon along with each positive or rated "
    "can be set using the " + PRINT_PARAM_STRING("neg") + "parameter. Also "
    "the number of epochs for which training is to be performed can be "
    "specified using the " + PRINT_PARAM_STRING("epochs") + "parameter."
    "\n\n"
    "A trained model may be saved to with the " +
    PRINT_PARAM_STRING("output_model") + " output parameter."
    "\n\n"
    "To train a NCF model on a dataset " + PRINT_DATASET("training_set") + " "
    "using GMF as algorithm using SGD optimizer and saving the trained model "
    "to " + PRINT_MODEL("model") + ", one could call: "
    "\n\n" +
    PRINT_CALL("ncf", "training", "training_set", "algorithm", "GMF",
        "output_model", "model", "optimizer", "SGD") +
    "\n\n"
    "Then, to use this model to generate recommendations for the list of users "
    "in the query set " + PRINT_DATASET("users") + ", storing 5 "
    "recommendations in " + PRINT_DATASET("recommendations") + ", one could "
    "call "
    "\n\n" +
    PRINT_CALL("ncf", "input_model", "model", "query", "users",
        "recommendations", 5, "output", "recommendations"));

// Parameters for training a model.
PARAM_MATRIX_IN("training", "Input dataset to perform CF on.", "t");
PARAM_STRING_IN("algorithm", "Algorithm for the network to be created.", "a",
    "GMF");
PARAM_STRING_IN("optimizer", "Optimizer to train the network on.", "z", "SGD");
PARAM_MATRIX_IN("test", "Test set to calculate RMSE on.", "T");

// Load/save a model.
PARAM_MODEL_IN(NCF, "input_model", "Trained NCF model to load.", "m");
PARAM_MODEL_OUT(NCF, "output_model", "Output for trained NCF model.", "M");

// Query settings.
PARAM_UMATRIX_IN("query", "List of query users for which recommendations should"
    " be generated.", "q");
PARAM_FLAG("all_user_recommendations", "Generate recommendations for all "
    "users.", "A");
PARAM_UMATRIX_OUT("output", "Matrix that will store output recommendations.",
    "o");
PARAM_INT_IN("recommendations", "Number of recommendations to generate for each"
    " query user.", "c", 5);
PARAM_INT_IN("embedsize", "Size of embedding to be used for each item and user "
    " data point.", "e", 8);
PARAM_INT_IN("neg", "Number of negative instances per positive instance to be "
    " trained upon.", "g", 4);
PARAM_INT_IN("epochs", "Number of epochs for which training is to be "
    " performed.", "p", 100);
PARAM_FLAG("implicit", "If true, treat the ratings as implicit feedback data."
    "i");

PARAM_INT_IN("seed", "Set the random seed (0 uses std::time(NULL)).", "s", 0);

void ComputeRecommendations(NCF* ncf,
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
    ncf->GetRecommendations(numRecs, recommendations, users.row(0).t());
  }
  else
  {
    Log::Info << "Generating recommendations for all users." << endl;
    ncf->GetRecommendations(numRecs, recommendations);
  }
}

void ComputeRMSE(NCF* ncf)
{
  // Now, compute each test point.
  arma::mat testData = std::move(CLI::GetParam<arma::mat>("test"));

  // Now compute the RMSE.
  size_t hitRatio, rmse;
  ncf->EvaluateModel(testData, hitRatio, rmse);

  Log::Info << "Hit Ratio is "<< hitRatio << "and RMSE is " << rmse << "." << endl;
}

void PerformAction(NCF* c)
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

  CLI::GetParam<NCF*>("output_model") = c;
}

template<typename OptimizerType>
void PerformAction(arma::mat& dataset,
                   std::string algorithm,
                   OptimizerType& optimizer)
{
  const size_t embedSize = (size_t) CLI::GetParam<int>("embedsize");
  const size_t neg = (size_t) CLI::GetParam<int>("neg");
  const size_t epochs = (size_t) CLI::GetParam<int>("epochs");
  const bool implicit = CLI::HasParam("implicit");

  NCF* c = new NCF(dataset, algorithm, optimizer, embedSize,
      neg, epochs, implicit);

  PerformAction(c);
}

void AssembleOptimizerType(const std::string& algorithm,
                           arma::mat& dataset)
{
  if (algorithm == "adagrad")
  {
    AdaGrad optimizer;
    PerformAction(dataset, algorithm, optimizer);
  }
  else if (algorithm == "rmsprop")
  {
    RMSProp optimizer;
    PerformAction(dataset, algorithm, optimizer);
  }
  else if (algorithm == "adam")
  {
    Adam optimizer;
    PerformAction(dataset, algorithm, optimizer);
  }
  else
  {
    SGD<> optimizer;
    PerformAction(dataset, algorithm, optimizer);
  }
}

static void mlpackMain()
{
  if (CLI::GetParam<int>("seed") == 0)
    math::RandomSeed(std::time(NULL));
  else
    math::RandomSeed(CLI::GetParam<int>("seed"));

  const string optimizerType = CLI::GetParam<string>("optimizer");

  // Validate parameters.
  RequireOnlyOnePassed({ "training", "input_model" }, true);

  // Check that nothing stupid is happening.
  if (CLI::HasParam("query") || CLI::HasParam("all_user_recommendations"))
    RequireOnlyOnePassed({ "query", "all_user_recommendations" }, true);

  RequireAtLeastOnePassed({ "output", "output_model" }, false,
      "no output will be saved");
  if (!CLI::HasParam("query") && !CLI::HasParam("all_user_recommendations"))
    ReportIgnoredParam("output", "no recommendations requested");

  RequireParamInSet<string>("algorithm", { "GMF", "MLP", "NeuMF" }, true,
      "unknown algorithm");

  RequireParamValue<int>("recommendations", [](int x) { return x > 0; }, true,
        "recommendations must be positive");

  // Either load from a model, or train a model.
  if (CLI::HasParam("training"))
  {
    // Train a model.

    // Read from the input file.
    arma::mat dataset = std::move(CLI::GetParam<arma::mat>("training"));

    // Recommendation matrix.
    arma::Mat<size_t> recommendations;

    const string algo = CLI::GetParam<string>("algorithm");

    // Perform the factorization and do whatever the user wanted.
    AssembleOptimizerType(algo, dataset);
  }
  else
  {
    // Load from a model after validating parameters.
    RequireAtLeastOnePassed({ "query", "all_user_recommendations",
        "test" }, true);

    // Load an input model.
    NCF* c = std::move(CLI::GetParam<NCF*>("input_model"));

    PerformAction(c);
  }
}
