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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_NAME("Linear Regression Predict");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of simple linear regression and ridge regression using.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of simple linear regression and simple ridge regression.");

// Example.
BINDING_EXAMPLE(
    "For example, to run a linear regression on the dataset.");

// See also...
BINDING_SEE_ALSO("Linear/ridge regression tutorial",
       "@doxygen/lrtutorial.html");

PARAM_MODEL_IN(LinearRegression, "input_model", "Existing LinearRegression "
    "model to use.", "m");

PARAM_MATRIX_IN("test", "Matrix containing X' (test regressors).", "T");

// This is the future name of the parameter.
PARAM_ROW_OUT("output_predictions", "If --test_file is specified, this "
    "matrix is where the predicted responses will be saved.", "o");

#ifndef MAKING_WRAPPER
static void mlpackMain()
{

  LinearRegression* lr;

  RequireAtLeastOnePassed({ "output_model", "output_predictions" }, false,
      "no output will be saved");

  // A model file was passed in, so load it.
  Timer::Start("load_model");
  lr = IO::GetParam<LinearRegression*>("input_model");
  Timer::Stop("load_model");

  // Cache the output of GetPrintableParam before we std::move() the test
  // matrix.  Loading actually will happen during GetPrintableParam() since
  // that needs to load to print the size.
  Timer::Start("load_test_points");
  std::ostringstream oss;
  oss << IO::GetPrintableParam<mat>("test");
  std::string testOutput = oss.str();
  Timer::Stop("load_test_points");

  mat points = std::move(IO::GetParam<mat>("test"));

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
  Timer::Start("prediction");
  lr->Predict(points, predictions);
  Timer::Stop("prediction");

  // Save predictions.
  IO::GetParam<rowvec>("output_predictions") = std::move(predictions);

}
#endif
