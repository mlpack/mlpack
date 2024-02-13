/**
 * @file methods/linear_regression/linear_regression_predict_main.cpp
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
#define BINDING_NAME linear_regression_predict

#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Linear Regression Prediction");

// Short description.
BINDING_SHORT_DESC("Predictions from model.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
  CALL_METHOD("model", "predict", "test", "X_test"));

PARAM_MODEL_IN_REQ(LinearRegression<>, "input_model", "Existing "
    "LinearRegression model to use.", "m");

PARAM_MATRIX_IN_REQ("test", "Matrix containing X' (test regressors).", "T");

// This is the future name of the parameter.
PARAM_ROW_OUT("output_predictions", "If --test_file is specified, this "
    "matrix is where the predicted responses will be saved.", "o");

void BINDING_FUNCTION(util::Params& params, util::Timers& timer)
{
  // A model file was passed in, so load it.
  timer.Start("load_model");
  LinearRegression<>* lr = params.Get<LinearRegression<>*>("input_model");
  timer.Stop("load_model");

  // Cache the output of GetPrintable before we std::move() the test
  // matrix.  Loading actually will happen during GetPrintable() since
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
