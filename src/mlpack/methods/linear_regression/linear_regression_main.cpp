/**
 * @file methods/linear_regression/linear_regression_main.cpp
 * @author James Cline
 *
 * Main function for least-squares linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME linear_regression

#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Simple Linear Regression and Prediction");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of simple linear regression and ridge regression using "
    "ordinary least squares.  Given a dataset and responses, a model can be "
    "trained and saved for later use, or a pre-trained model can be used to "
    "output regression predictions for a test set.");

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
    "invertible.  The calculated b may be saved with the " +
    PRINT_PARAM_STRING("output_predictions") + " output parameter."
    "\n\n"
    "Optionally, the calculated value of b is used to predict the responses for"
    " another matrix X' (specified by the " + PRINT_PARAM_STRING("test") + " "
    "parameter):"
    "\n\n"
    "   y' = X' * b"
    "\n\n"
    "and the predicted responses y' may be saved with the " +
    PRINT_PARAM_STRING("output_predictions") + " output parameter.  This type "
    "of regression is related to least-angle regression, which mlpack "
    "implements as the 'lars' program.");

// Example.
BINDING_EXAMPLE(
    "For example, to run a linear regression on the dataset " +
    PRINT_DATASET("X") + " with responses " + PRINT_DATASET("y") + ", saving "
    "the trained model to " + PRINT_MODEL("lr_model") + ", the following "
    "command could be used:"
    "\n\n" +
    PRINT_CALL("linear_regression", "training", "X", "training_responses", "y",
        "output_model", "lr_model") +
    "\n\n"
    "Then, to use " + PRINT_MODEL("lr_model") + " to predict responses for a "
    "test set " + PRINT_DATASET("X_test") + ", saving the predictions to " +
    PRINT_DATASET("X_test_responses") + ", the following command could be "
    "used:"
    "\n\n" +
    PRINT_CALL("linear_regression", "input_model", "lr_model", "test", "X_test",
        "output_predictions", "X_test_responses"));

// See also...
BINDING_SEE_ALSO("@lars", "#lars");
BINDING_SEE_ALSO("Linear regression on Wikipedia",
    "https://en.wikipedia.org/wiki/Linear_regression");
BINDING_SEE_ALSO("LinearRegression C++ class documentation",
    "@doc/user/methods/linear_regression.md");

PARAM_MATRIX_IN("training", "Matrix containing training set X (regressors).",
    "t");
PARAM_ROW_IN("training_responses", "Optional vector containing y "
    "(responses). If not given, the responses are assumed to be the last row "
    "of the input file.", "r");

PARAM_MODEL_IN(LinearRegression<>, "input_model", "Existing LinearRegression "
    "model to use.", "m");
PARAM_MODEL_OUT(LinearRegression<>, "output_model", "Output LinearRegression "
    "model.", "M");

PARAM_MATRIX_IN("test", "Matrix containing X' (test regressors).", "T");

// This is the future name of the parameter.
PARAM_ROW_OUT("output_predictions", "If --test_file is specified, this "
    "matrix is where the predicted responses will be saved.", "o");

PARAM_DOUBLE_IN("lambda", "Tikhonov regularization for ridge regression.  If 0,"
    " the method reduces to linear regression.", "l", 0.0);

void BINDING_FUNCTION(util::Params& params, util::Timers& timer)
{
  const double lambda = params.Get<double>("lambda");

  RequireOnlyOnePassed(params, { "training", "input_model" }, true);

  ReportIgnoredParam(params, {{ "test", false }}, "output_predictions");

  mat regressors;
  rowvec responses;

  LinearRegression<>* lr;

  const bool computeModel = !params.Has("input_model");
  const bool computePrediction = params.Has("test");

  // If they specified a model file, we also need a test file or we
  // have nothing to do.
  if (!computeModel)
  {
    RequireAtLeastOnePassed(params, { "test" }, true, "test points must be "
        "specified when an input model is given");
  }

  ReportIgnoredParam(params, {{ "input_model", true }}, "lambda");

  RequireAtLeastOnePassed(params, { "output_model", "output_predictions" },
      false, "no output will be saved");

  // An input file was given and we need to generate the model.
  if (computeModel)
  {
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

    timer.Start("regression");
    lr = new LinearRegression<>(regressors, responses, lambda);
    timer.Stop("regression");
  }
  else
  {
    // A model file was passed in, so load it.
    timer.Start("load_model");
    lr = params.Get<LinearRegression<>*>("input_model");
    timer.Stop("load_model");
  }

  // Did we want to predict, too?
  if (computePrediction)
  {
    // Cache the output of GetPrintableParam before we std::move() the test
    // matrix.  Loading actually will happen during GetPrintableParam() since
    // that needs to load to print the size.
    timer.Start("load_test_points");
    std::ostringstream oss;
    oss << params.GetPrintable<mat>("test");
    std::string testOutput = oss.str();
    timer.Stop("load_test_points");

    mat points = std::move(params.Get<mat>("test"));

    // Ensure that test file data has the right number of features.
    if ((lr->Parameters().n_elem - 1) != points.n_rows)
    {
      // If we built the model, nothing will free it so we have to...
      const size_t dimensions = lr->Parameters().n_elem - 1;
      if (computeModel)
        delete lr;

      Log::Fatal << "The model was trained on " << dimensions << "-dimensional "
          << "data, but the test points in '" << testOutput << "' are "
          << points.n_rows << "-dimensional!" << endl;
    }

    // Perform the predictions using our model.
    rowvec predictions;
    timer.Start("prediction");
    lr->Predict(points, predictions);
    timer.Stop("prediction");

    // Save predictions.
    params.Get<rowvec>("output_predictions") = std::move(predictions);
  }

  // Save the model if needed.
  params.Get<LinearRegression<>*>("output_model") = lr;
}
