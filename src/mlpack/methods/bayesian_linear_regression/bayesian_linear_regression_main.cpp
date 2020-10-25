/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression_main.cpp
 * @author Clement Mercier
 *
 * Executable for BayesianLinearRegression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "bayesian_linear_regression.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;

// Program Name.
BINDING_NAME("BayesianLinearRegression");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the bayesian linear regression.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of the bayesian linear regression."
    "\n"
    "This model is a probabilistic view and implementation of the linear "
    "regression. The final solution is obtained by computing a posterior "
    "distribution from gaussian likelihood and a zero mean gaussian isotropic "
    " prior distribution on the solution. "
    "\n"
    "Optimization is AUTOMATIC and does not require cross validation. "
    "The optimization is performed by maximization of the evidence function. "
    "Parameters are tuned during the maximization of the marginal likelihood. "
    "This procedure includes the Ockham's razor that penalizes over complex "
    "solutions. "
    "\n\n"
    "This program is able to train a Bayesian linear regression model or load "
    "a model from file, output regression predictions for a test set, and save "
    "the trained model to a file."
    "\n\n"
    "To train a BayesianLinearRegression model, the " +
    PRINT_PARAM_STRING("input") + " and " + PRINT_PARAM_STRING("responses") +
    "parameters must be given. The " + PRINT_PARAM_STRING("center") +
    "and " + PRINT_PARAM_STRING("scale") + " parameters control the "
    "centering and the normalizing options. A trained model can be saved with "
    "the " + PRINT_PARAM_STRING("output_model") + ". If no training is desired "
    "at all, a model can be passed via the " +
    PRINT_PARAM_STRING("input_model") + " parameter."
    "\n\n"
    "The program can also provide predictions for test data using either the "
    "trained model or the given input model.  Test points can be specified "
    "with the " + PRINT_PARAM_STRING("test") + " parameter.  Predicted "
    "responses to the test points can be saved with the " +
    PRINT_PARAM_STRING("predictions") + " output parameter. The "
    "corresponding standard deviation can be save by precising the " +
    PRINT_PARAM_STRING("stds") + " parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, the following command trains a model on the data " +
    PRINT_DATASET("data") + " and responses " + PRINT_DATASET("responses") +
    "with center set to true and scale set to false (so, Bayesian "
    "linear regression is being solved, and then the model is saved to " +
    PRINT_MODEL("blr_model") + ":"
    "\n\n" +
    PRINT_CALL("bayesian_linear_regression", "input", "data", "responses",
               "responses", "center", 1, "scale", 0, "output_model",
               "blr_model") +
    "\n\n"
    "The following command uses the " +
    PRINT_MODEL("blr_model") + " to provide predicted " +
    " responses for the data " + PRINT_DATASET("test") + " and save those " +
    " responses to " + PRINT_DATASET("test_predictions") + ": "
    "\n\n" +
    PRINT_CALL("bayesian_linear_regression", "input_model",
               "blr_model", "test", "test",
               "predictions", "test_predictions") +
    "\n\n"
    "Because the estimator computes a predictive distribution instead of "
    "a simple point estimate, the " + PRINT_PARAM_STRING("stds") + " parameter "
    "allows one to save the prediction uncertainties: "
    "\n\n" +
    PRINT_CALL("bayesian_linear_regression", "input_model",
               "blr_model", "test", "test",
               "predictions", "test_predictions", "stds", "stds"));

// See also...
BINDING_SEE_ALSO("Bayesian Interpolation",
        "https://authors.library.caltech.edu/13792/1/MACnc92a.pdf");
BINDING_SEE_ALSO("Bayesian Linear Regression, Section 3.3",
        "MLA Bishop, Christopher M. Pattern Recognition and Machine "
        "Learning. New York :Springer, 2006, section 3.3.");
BINDING_SEE_ALSO("mlpack::regression::BayesianLinearRegression C++ class "
        "documentation",
        "@doxygen/classmlpack_1_1regression_1_1BayesianLinearRegression.html");

PARAM_MATRIX_IN("input", "Matrix of covariates (X).", "i");

PARAM_ROW_IN("responses", "Matrix of responses/observations (y).", "r");

PARAM_MODEL_IN(BayesianLinearRegression, "input_model", "Trained "
               "BayesianLinearRegression model to use.", "m");

PARAM_MODEL_OUT(BayesianLinearRegression, "output_model", "Output "
                "BayesianLinearRegression model.", "M");

PARAM_MATRIX_IN("test", "Matrix containing points to regress on (test "
                "points).", "t");

PARAM_MATRIX_OUT("predictions", "If --test_file is specified, this "
                  "file is where the predicted responses will be saved.", "o");

PARAM_MATRIX_OUT("stds", "If specified, this is where the standard deviations "
    "of the predictive distribution will be saved.", "u");

PARAM_FLAG("center", "Center the data and fit the intercept if enabled.", "c");

PARAM_FLAG("scale", "Scale each feature by their standard deviations if "
           "enabled.", "s");

static void mlpackMain()
{
  bool center = IO::GetParam<bool>("center");
  bool scale = IO::GetParam<bool>("scale");

  // Check parameters -- make sure everything given makes sense.
  RequireOnlyOnePassed({"input", "input_model"}, true);
  if (IO::HasParam("input"))
  {
    RequireOnlyOnePassed({"responses"}, true, "if input data is specified, "
        "responses must also be specified");
  }
  ReportIgnoredParam({{"input", false }}, "responses");

  RequireAtLeastOnePassed({"predictions", "output_model", "stds"}, false,
      "no results will be saved");

  // Ignore out_predictions unless test is specified.
  ReportIgnoredParam({{"test", false}}, "predictions");

  BayesianLinearRegression* bayesLinReg;
  if (IO::HasParam("input"))
  {
    Log::Info << "input detected " << std::endl;
    // Initialize the object.
    bayesLinReg = new BayesianLinearRegression(center, scale);

    // Load covariates.  We can avoid LARS transposing our data by choosing to
    // not transpose this data (that's why we used PARAM_TMATRIX_IN).
    mat matX = std::move(IO::GetParam<arma::mat>("input"));

    // Load responses.  The responses should be a one-dimensional vector, and it
    // seems more likely that these will be stored with one response per line
    // (one per row). So we should not transpose upon loading.
    arma::rowvec responses = std::move(
        IO::GetParam<arma::rowvec>("responses"));

    if (responses.n_elem != matX.n_cols)
    {
      delete bayesLinReg;
      Log::Fatal << "Number of responses must be equal to number of rows of X!"
                 << endl;
    }

    arma::rowvec predictionsTrain;
    // The Train method is ready to take data in column-major format.
    bayesLinReg->Train(matX, responses);
  }
  else // We must have --input_model_file.
  {
    bayesLinReg = IO::GetParam<BayesianLinearRegression*>("input_model");
  }

  if (IO::HasParam("test"))
  {
    Log::Info << "Regressing on test points." << endl;
    // Load test points.
    mat testPoints = std::move(IO::GetParam<arma::mat>("test"));
    arma::rowvec predictions;

    if (IO::HasParam("stds"))
    {
      arma::rowvec std;
      bayesLinReg->Predict(testPoints, predictions, std);

      // Save the standard deviation of the test points (one per line).
      IO::GetParam<arma::mat>("stds") = std::move(std);
    }
    else
    {
      bayesLinReg->Predict(testPoints, predictions);
    }

    // Save test predictions (one per line).
    IO::GetParam<arma::mat>("predictions") = std::move(predictions);
  }

  IO::GetParam<BayesianLinearRegression*>("output_model") = bayesLinReg;
}
