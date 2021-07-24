/**
 * @file methods/linear_regression/linear_regression_fit_main.cpp
 * @author James Cline
 *
 * Main function for least-squares linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME linear_regression_fit

#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Simple Linear Regression Training");

// Short description.
BINDING_SHORT_DESC(
  "An implementation of simple linear regression and ridge regression using "
  "ordinary least squares.  Given a dataset and responses, a model can be "
  "trained and saved for later use.");

// Long description.
BINDING_LONG_DESC(
  "An implementation of simple linear regression and simple ridge regression "
  "using ordinary least squares. This solves the problem"
  "\n\n"
  "  y = X * b + e"
  "\n\n"
  "where X (specified by " + PRINT_PARAM_STRING("training") + ") and y "
  "(specified either as the last column of the input matrix " +
  PRINT_PARAM_STRING("training") + " or via the " +
  PRINT_PARAM_STRING("training_responses") + " parameter) are known and b is"
  " the desired variable.  If the covariance matrix (X'X) is not invertible, "
  "or if the solution is overdetermined, then specify a Tikhonov "
  "regularization constant (with " + PRINT_PARAM_STRING("lambda") + ") "
  "greater than 0, which will regularize the covariance matrix to make it "
  "invertible. The calculated value of b can be used to predict through the "
  "predict program of linear regression.");

// Example.
BINDING_EXAMPLE(
  "For example, to fit a linear regression on the dataset ");

// See also...
BINDING_SEE_ALSO("Linear/ridge regression tutorial",
       "@doxygen/lrtutorial.html");

PARAM_MATRIX_IN("training", "Matrix containing training set X (regressors).",
    "t");
PARAM_ROW_IN("training_responses", "Optional vector containing y "
    "(responses). If not given, the responses are assumed to be the last row "
    "of the input file.", "r");

PARAM_MODEL_OUT(LinearRegression, "output_model", "Output LinearRegression "
    "model.", "M");

PARAM_DOUBLE_IN("lambda", "Tikhonov regularization for ridge regression.  If 0,"
    " the method reduces to linear regression.", "l", 0.0);

void BINDING_FUNCTION(util::Params& params, util::Timers& timer)
{
  const double lambda = params.Get<double>("lambda");
  RequireOnlyOnePassed(params, {"training"}, true); // training must be passsed.

  mat regressors;
  rowvec responses;

  LinearRegression* lr;

  timer.Start("load_regressors");
  regressors = std::move(params.Get<mat>("training"));
  timer.Stop("load_regressors");

  // Are the responses in a separate file?
  if (!params.Has("training_responses"))
  {
    // The initial predictors for y, Nx1.
    if (regressors.n_rows < 2)
    {
      Log::Fatal << "Can't get responses from training data "
          "since it has less than 2 rows." << endl;
    }
    responses = regressors.row(regressors.n_rows - 1);
    regressors.shed_row(regressors.n_rows - 1);
  }
  else
  {
    // The initial predictors for y, Nx1.
    timer.Start("load_responses");
    responses = params.Get<rowvec>("training_responses");
    timer.Stop("load_responses");

    if (responses.n_cols != regressors.n_cols)
    {
      Log::Fatal << "The responses must have the same number of columns "
          "as the training set." << endl;
    }
  }

  if(regressors.n_cols != responses.n_cols)
    Log::Fatal << "Regressors and Responses must have the same number of data points!" << endl;

  timer.Start("regression");
  lr = new LinearRegression(regressors, responses, lambda);
  timer.Stop("regression");

  // Save the model if needed.
  params.Get<LinearRegression*>("output_model") = lr;
}
