/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression_predict_main.cpp
 * @author Clement Mercier
 * @author Dirk Eddelbuettel
 *
 * Executable for BayesianLinearRegression prediction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME bayesian_linear_regression_predict

#include <mlpack/core/util/mlpack_main.hpp>

#include "bayesian_linear_regression.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("BayesianLinearRegression Prediction");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the Bayesian linear regression prediction: Given a "
    "pre-trained model and a test data set, it provides model predictions.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "predict", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("Bayesian Interpolation",
    "https://cs.uwaterloo.ca/~mannr/cs886-w10/mackay-bayesian.pdf");
BINDING_SEE_ALSO("Bayesian Linear Regression, Section 3.3",
    // I wonder how long this full text PDF will remain available...
    "https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/"
    "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf");
BINDING_SEE_ALSO("BayesianLinearRegression C++ class documentation",
    "@doc/user/methods/bayesian_linear_regression.md");

PARAM_MODEL_IN_REQ(BayesianLinearRegression<>, "input_model", "Trained "
    "BayesianLinearRegression model to use.", "m");

PARAM_MATRIX_IN_REQ("test", "Matrix containing points to regress on (test "
    "points).", "t");

PARAM_FLAG("stddevs", "Return standard deviations along with predictions.",
           "s");

PARAM_MATRIX_OUT("predictions", "Matrix of predicted responses, with "
    "associated standard deviations if option selected.", "o");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  BayesianLinearRegression<>* bayesLinReg;
  bayesLinReg = params.Get<BayesianLinearRegression<>*>("input_model");
  bool stddevs = params.Get<bool>("stddevs");

  // Load test points.
  mat testPoints = std::move(params.Get<arma::mat>("test"));

  if (stddevs)
  {
    arma::rowvec predictions;
    arma::rowvec std;

    timers.Start("bayesian_linear_regression_prediction");
    bayesLinReg->Predict(testPoints, predictions, std);
    timers.Stop("bayesian_linear_regression_prediction");

    arma::mat res = arma::join_cols(predictions, std);
    // Save test predictions and standard deviations (one per line).
    params.Get<arma::mat>("predictions") = std::move(res);
  }
  else
  {
    arma::rowvec predictions;
    timers.Start("bayesian_linear_regression_prediction");
    bayesLinReg->Predict(testPoints, predictions);
    timers.Stop("bayesian_linear_regression_prediction");
    // Save test predictions and standard deviations (two lines).
    params.Get<arma::mat>("predictions") = std::move(predictions);
  }
}
