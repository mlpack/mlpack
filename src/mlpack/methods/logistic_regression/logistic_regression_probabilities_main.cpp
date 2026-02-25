/**
 * @file methods/logistic_regression/logistic_regression_probabilities_main.cpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * Main executable for logistic regression probabilities step.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME logistic_regression_probabilities

#include <mlpack/core/util/mlpack_main.hpp>

#include "logistic_regression.hpp"
#include "logistic_regression_function.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("L2-regularized Logistic Regression Probabilities");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of L2-regularized logistic regression for two-class "
    "classification.  Uses a trained model to classify new points and provide "
    "classification probabilties.");

// Long description. Taken from logistic_regression_train_main.cpp
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "probabilities", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@logistic_regression_train", "#logistic_regression_train");
BINDING_SEE_ALSO("@logistic_regression_classify",
    "#logistic_regression_classify");
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Logistic regression on Wikipedia",
    "https://en.wikipedia.org/wiki/Logistic_regression");
BINDING_SEE_ALSO(":LogisticRegression C++ class documentation",
    "@doc/user/methods/logistic_regression.md");

// Classification options.
PARAM_MATRIX_IN_REQ("test", "Matrix containing test dataset.", "T");
PARAM_DOUBLE_IN("decision_boundary", "Decision boundary for prediction; if the "
    "logistic function for a point is less than the boundary, the class is "
    "taken to be 0; otherwise, the class is 1.", "d", 0.5);
PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each "
    "point in the test set.", "p");

// Model loading/saving.
PARAM_MODEL_IN_REQ(LogisticRegression<>, "input_model", "Existing model "
    "(parameters).", "m");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  LogisticRegression<>* model =
    params.Get<LogisticRegression<>*>("input_model");
  const arma::mat& testSet = params.Get<arma::mat>("test");

  // Checking the dimensionality of the test data.
  if (testSet.n_rows != model->Parameters().n_cols - 1)
  {
    // Clean memory if needed.
    const size_t trainingDimensionality = model->Parameters().n_cols - 1;
    if (!params.Has("input_model"))
      delete model;

    Log::Fatal << "Test data dimensionality (" << testSet.n_rows << ") must "
        << "be the same as the dimensionality of the training data ("
        << trainingDimensionality << ")!" << endl;
  }

  // Decision boundary must be between 0 and 1.
  RequireParamValue<double>(params, "decision_boundary",
      [](double x) { return x >= 0.0 && x <= 1.0; }, true,
      "decision boundary must be between 0.0 and 1.0");

  // We must perform predictions on the test set.  Training (and the
  // optimizer) are irrelevant here; we'll pass in the model we have.
  arma::Row<size_t> predictions;
  arma::mat probabilities;

  Log::Info << "Predicting classes of points in '"
      << params.GetPrintable<arma::mat>("test") << "'." << endl;
  timers.Start("logistic_regression_probabilities");
  model->Classify(testSet, predictions, probabilities);
  timers.Stop("logistic_regression_probabilities");

  params.Get<arma::mat>("probabilities") = std::move(probabilities);
}
