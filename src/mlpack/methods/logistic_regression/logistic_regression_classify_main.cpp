/**
 * @file methods/logistic_regression/logistic_regression_classify_main.cpp
 * @author Ryan Curtin
 *
 * Main executable for logistic regression classification step.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME logistic_regression_classify

#include <mlpack/core/util/mlpack_main.hpp>

#include "logistic_regression.hpp"
#include "logistic_regression_function.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("L2-regularized Logistic Regression Classification");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of L2-regularized logistic regression for two-class "
    "classification.  Given labeled data, a trained model can be used to "
    "classify new points.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of L2-regularized logistic regression for two-class "
    "classification.  Given labeled data, a trained model can be used to "
    "classify new points.");
    // TODO:FIXME:EXTEND ?

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "classify", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@logistic_regression_train", "#logistic_regression_train");
BINDING_SEE_ALSO("@logistic_regression_probabilities", "#logistic_regression_probabilities");
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
PARAM_UROW_OUT("predictions", "If test data is specified, this matrix is where "
    "the predictions for the test set will be saved.", "P");

// Model loading/saving.
PARAM_MODEL_IN_REQ(LogisticRegression<>, "input_model", "Existing model "
    "(parameters).", "m");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  LogisticRegression<>* model = params.Get<LogisticRegression<>*>("input_model");
  const double decisionBoundary = params.Get<double>("decision_boundary");

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
  Log::Info << "Predicting classes of points in '"
      << params.GetPrintable<arma::mat>("test") << "'." << endl;
  timers.Start("logistic_regression_classification");
  model->Classify(testSet, predictions, decisionBoundary);
  timers.Stop("logistic_regression_classification");

  params.Get<arma::Row<size_t>>("predictions") = predictions;
}
