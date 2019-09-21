/**
 * @file bayesian_ridge_main.cpp
 * @author Clement Mercier
 *
 * Executable for BayesianRidge.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "bayesian_ridge.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;

PROGRAM_INFO("BayesianRidge",
    // Short description.
    " An implementation of the bayesian linear regression, also known"
    "as the Bayesian Ridge regression. This can train a Bayesian Ridge model "
    "and use that model or a pre-trained model to output regression predictions "
    "for a test set.",
    // Long description.
    "An implementation of the bayesian linear regression, also known"
    "as the Bayesian Ridge regression.\n "
    "This is a probabilistic view and implementation of the Ridge regression. "
    "Final solution is obtained by comptuting a posterior distribution from "
    "gaussian likelihood and a gaussian isotropic prior distribution on the "
    "weigths. "
    "\n"
    "Optimization is AUTOMATIC and does not require cross validation. "
    "The optimization is performed by type II maximium likihood. Parameters "
    "are tunned during the maximization of the marginal likelihood. This "
    "procedure includes the Occam's razor that penalizes over complex solutions. "
    "\n\n"
    "This program is able to train a Baysian Ridge model or load a "
    "model from file, output regression predictions for a test set, and save "
    "the trained model to a file.  The Bayesian Ridge algorithm is described in more "
    "detail below:"
    "\n\n"
    "Let X be a matrix where each row is a point and each column is a "
    "dimension, t is a vector of targets, alpha is the precision of the "
    "gaussian prior distribtion of w, and w is solution to compute. "
    "\n\n"
    "The Bayesian Ridge comptute the posterior distribution of the parameters "
    "by the Bayes's rule : "
    "\n\n"
    " p(w|X) = p(X,t|w) * p(w|alpha) / p(X)"
    "\n\n"
    "To train a BayesianRidge model, the " +
    PRINT_PARAM_STRING("input") + " and " + PRINT_PARAM_STRING("responses") +
    "parameters must be given. The " + PRINT_PARAM_STRING("fitIntercept") +
    "and " + PRINT_PARAM_STRING("normalize") + " parameters control the "
    "centering and the normalizing options. A trained model can be saved with "
    "the " + PRINT_PARAM_STRING("output_model") + ". If no training is desired "
    "at all, a model can be passed via the "+ PRINT_PARAM_STRING("input_model") +
    " parameter."
    "\n\n"
    "The program can also provide predictions for test data using either the "
    "trained model or the given input model.  Test points can be specified with"
    " the " + PRINT_PARAM_STRING("test") + " parameter.  Predicted responses "
    "to the test points can be saved with the " +
    PRINT_PARAM_STRING("output_predictions") + " output parameter."
    "\n\n"
    "For example, the following command trains a model on the data " +
    PRINT_DATASET("data") + " and responses " + PRINT_DATASET("responses") +
    " with fitIntercept set to true and normalize set to false (so, Bayesian Ridge "
    "is being solved, and then the model is saved to " +
    PRINT_MODEL("bayesian_ridge_model") + ":"
    "\n\n" +
    PRINT_CALL("bayesian_ridge", "input", "data", "responses", "responses",
	       "fitIntercept", 1, "normalize", 0, "output_model",
	       "bayesian_ridge_model") +
    "\n\n"
    "The following command uses the " + PRINT_MODEL("bayesian_ridge_model") +
    " to provide predicted responses for the data " + PRINT_DATASET("test") +  
    " and save those responses to " + PRINT_DATASET("test_predictions") + ": "
    "\n\n" +
    PRINT_CALL("bayesian_ridge", "input_model", "bayesian_ridge_model", "test",
    	       "test", "output_predictions", "test_predictions"));

PARAM_TMATRIX_IN("input", "Matrix of covariates (X).", "i");
PARAM_MATRIX_IN("responses", "Matrix of responses/observations (y).", "r");

PARAM_MODEL_IN(BayesianRidge, "input_model", "Trained LARS model to use.", "m");
PARAM_MODEL_OUT(BayesianRidge, "output_model", "Output LARS model.", "M");

PARAM_TMATRIX_IN("test", "Matrix containing points to regress on (test "
    "points).", "t");

PARAM_TMATRIX_OUT("output_predictions", "If --test_file is specified, this "
    "file is where the predicted responses will be saved.", "o");

PARAM_INT_IN("fitIntercept", "Center the data and fit the intercept",
	     "f",
	     1);
PARAM_INT_IN("normalize", "Normlize each feature by their standard deviations.",
	     "n",
	     0);

static void mlpackMain()
{
  int fitIntercept = CLI::GetParam<int>("fitIntercept");
  int normalize = CLI::GetParam<int>("normalize");

  // Check parameters -- make sure everything given makes sense.
  RequireOnlyOnePassed({ "input", "input_model" }, true);
  if (CLI::HasParam("input"))
  {
    RequireOnlyOnePassed({ "responses" }, true, "if input data is specified, "
        "responses must also be specified");
  }
  ReportIgnoredParam({{"input", false }}, "responses");

  RequireAtLeastOnePassed({ "output_predictions", "output_model" }, false,
      "no results will be saved");
  // Is this line really rigth ? It comes from lars_main.hpp.
  // ReportIgnoredParam({{ "test", true }}, "output_predictions");

  BayesianRidge* bayesRidge;
  if (CLI::HasParam("input"))
  {
    Log::Info << "input detected " << std::endl; 
    // Initialize the object.
    bayesRidge = new BayesianRidge(fitIntercept, normalize);

    // Load covariates.  
    mat matX = std::move(CLI::GetParam<arma::mat>("input"));

    // Load responses.  The responses should be a one-dimensional vector, and it
    // seems more likely that these will be stored with one response per line
    // (one per row).  So we should not transpose upon loading.
    mat matY = std::move(CLI::GetParam<arma::mat>("responses"));

    // Make sure y is oriented the right way.
    if (matY.n_cols == 1)
      matY = trans(matY);
    if (matY.n_rows > 1)
      Log::Fatal << "Only one column or row allowed in responses file!" << endl;

    if (matY.n_elem != matX.n_rows)
      Log::Fatal << "Number of responses must be equal to number of rows of X!"
          << endl;

    arma::rowvec y = std::move(matY);
    arma::rowvec predictionsTrain;
    // The Train method is ready to take data in colomn-major format.
    bayesRidge->Train(matX.t(), matY);
  }
  else // We must have --input_model_file.
  {
    bayesRidge = CLI::GetParam<BayesianRidge*>("input_model");
  }

  if (CLI::HasParam("test"))
  {
    Log::Info << "Regressing on test points." << endl;
    // Load test points.
    mat testPoints = std::move(CLI::GetParam<arma::mat>("test"));

    arma::rowvec predictions;
    bayesRidge->Predict(testPoints.t(), predictions);

    // Save test predictions (one per line).
    CLI::GetParam<arma::mat>("output_predictions") = std::move(predictions.t());
    Log::Info << predictions << std::endl;
  }
  
  CLI::GetParam<BayesianRidge*>("output_model") = bayesRidge;
}
