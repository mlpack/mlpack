/**
 * @file methods/linear_regression/linear_regression_train_main.cpp
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

#undef BINDING_NAME
#define BINDING_NAME linear_regression_train

#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Simple Linear Regression");

// Short description.
BINDING_SHORT_DESC(
  "Train a linear regression model.");

// Long description.
BINDING_LONG_DESC(
  "An implementation of simple linear regression and simple ridge regression "
  "using ordinary least squares. This solves the problem"
  "\n\n"
  "  y = X * b + e");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("linear_regression") + "\n" +
    GET_DATASET("X", "https://example.com") + "\n" +
    GET_DATASET("y", "https://example.com") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
        "0.2") + "\n" +
    CREATE_OBJECT("lr", "linear_regression") + "\n" +
    CALL_METHOD("lr", "train", "training", "X_train", "training_responses",
        "y_train"));

// See also...
PARAM_MATRIX_IN_REQ("training", "Matrix containing training set X "
    "(regressors).", "t");
PARAM_ROW_IN("training_responses", "Optional vector containing y "
    "(responses). If not given, the responses are assumed to be the last row "
    "of the input file.", "r");

PARAM_MODEL_OUT(LinearRegression<>, "output_model", "Output LinearRegression "
    "model.", "M");

PARAM_DOUBLE_IN("lambda", "Tikhonov regularization for ridge regression.  If 0,"
    " the method reduces to linear regression.", "l", 0.0);

void BINDING_FUNCTION(util::Params& params, util::Timers& timer)
{
  const double lambda = params.Get<double>("lambda");

  mat regressors;
  rowvec responses;

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

  if (regressors.n_cols != responses.n_cols)
    Log::Fatal << "Regressors and responses must have the same number of data "
        << "points!" << endl;

  timer.Start("regression");
  LinearRegression<>* lr = new LinearRegression<>(regressors, responses,
      lambda);
  timer.Stop("regression");

  // Save the model if needed.
  params.Get<LinearRegression<>*>("output_model") = lr;
}
