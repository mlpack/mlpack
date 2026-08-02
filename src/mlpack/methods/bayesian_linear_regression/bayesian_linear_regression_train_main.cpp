/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression_train_main.cpp
 * @author Clement Mercier
 * @author Dirk Eddelbuettel
 *
 * Executable for BayesianLinearRegression Training (i.e Fit).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME bayesian_linear_regression_train

#include <mlpack/core/util/mlpack_main.hpp>

#include "bayesian_linear_regression.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("BayesianLinearRegression Training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the Bayesian linear regression training.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of the Bayesian linear regression."
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
    "To train a BayesianLinearRegression model, the " +
    PRINT_PARAM_STRING("input") + " and " + PRINT_PARAM_STRING("responses") +
    " parameters must be given. The " + PRINT_PARAM_STRING("center") +
    " and " + PRINT_PARAM_STRING("scale") + " parameters control the "
    "centering and the normalizing options. A trained model is returned."
    "\n\n");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("bayesian_linear_regression", "train", "predict") + "\n" +
    GET_DATASET("X",
        "http://datasets.mlpack.org/admission_predict.csv") + "\n" +
    GET_DATASET("y",
        "http://datasets.mlpack.org/admission_predict.responses.csv") + "\n" +
    SPLIT_TRAIN_TEST_REGRESSION("X", "y", "X_train", "y_train", "X_test",
        "y_test", "0.2") + "\n" +
    CREATE_OBJECT("model", "bayesian_linear_regression") + "\n" +
    CALL_METHOD("model", "train", "input", "X_train", "responses", "y_train",
      "center", 1, "scale", 0, "output_model",
      "bayesian_linear_regression_model"));

// See also...
BINDING_SEE_ALSO("Bayesian Interpolation",
    "https://cs.uwaterloo.ca/~mannr/cs886-w10/mackay-bayesian.pdf");
BINDING_SEE_ALSO("Bayesian Linear Regression, Section 3.3",
    // I wonder how long this full text PDF will remain available...
    "https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/"
    "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf");
BINDING_SEE_ALSO("BayesianLinearRegression C++ class documentation",
    "@doc/user/methods/bayesian_linear_regression.md");

PARAM_MATRIX_IN_REQ("input", "Matrix of covariates (X).", "i");

PARAM_ROW_IN_REQ("responses", "Matrix of responses/observations (y).", "r");

PARAM_MODEL_OUT(BayesianLinearRegression<>, "output_model", "Output "
                "BayesianLinearRegression model.", "M");

PARAM_FLAG("center", "Center the data and fit the intercept if enabled.", "c");

PARAM_FLAG("scale", "Scale each feature by their standard deviations if "
           "enabled.", "s");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  bool center = params.Get<bool>("center");
  bool scale = params.Get<bool>("scale");

  BayesianLinearRegression<>* bayesLinReg;
  // Initialize the object.
  bayesLinReg = new BayesianLinearRegression<>(center, scale);

  // Load covariates.
  mat matX = std::move(params.Get<arma::mat>("input"));

  // Load responses.  The responses should be a one-dimensional vector, and it
  // seems more likely that these will be stored with one response per line
  // (one per row). So we should not transpose upon loading.
  arma::rowvec responses = std::move(params.Get<arma::rowvec>("responses"));

  if (responses.n_elem != matX.n_cols)
  {
    delete bayesLinReg;
    Log::Fatal << "Number of responses must be equal to number of rows of X!"
               << endl;
  }

  // The Train method is ready to take data in column-major format.
  timers.Start("bayesian_linear_regression_training");
  bayesLinReg->Train(matX, responses);
  timers.Stop("bayesian_linear_regression_training");

  params.Get<BayesianLinearRegression<>*>("output_model") = bayesLinReg;
}
