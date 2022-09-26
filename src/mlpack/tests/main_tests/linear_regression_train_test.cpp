/**
 * @file tests/main_tests/linear_regression_train_test.cpp
 * @author Nippun Sharma
 *
 * Test RUN_BINDING() of linear_regression_fit_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LRFitTestFixture);

/**
 * Training a model with different regularization parameter and ensuring that
 * predictions are different.
 */
TEST_CASE_METHOD(LRFitTestFixture, "LRFitDifferentLambdas",
                 "[LinearRegressionFitMainTest][BindingTests]")
{
  // A required minimal difference between solutions.
  const double delta = 0.1;

  arma::mat trainX({1.0, 2.0, 3.0});
  arma::mat testX({4.0});
  arma::rowvec trainY({1.0, 4.0, 9.0});

  SetInputParam("training", trainX);
  SetInputParam("training_responses", trainY);
  SetInputParam("lambda", 0.1);

  // The first solution.
  RUN_BINDING();
  arma::rowvec preds1;
  params.Get<LinearRegression*>("output_model")->Predict(testX,
      preds1);
  const double testY1 = preds1(0);

  ResetSettings();

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("lambda", 1.0);

  // The second solution.
  RUN_BINDING();
  arma::rowvec preds2;
  params.Get<LinearRegression*>("output_model")->Predict(testX,
      preds2);
  const double testY2 = preds2(0);

  // Second solution has stronger regularization,
  // so the predicted value should be smaller.
  REQUIRE(testY1 - delta > testY2);
}

/**
 * Ensuring that response size is checked.
 */
TEST_CASE_METHOD(LRFitTestFixture, "LRFitWrongResponseSizeTest",
                 "[LinearRegressionFitMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N + 3); // Wrong size.

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that absence of responses is checked.
 */
TEST_CASE_METHOD(LRFitTestFixture, "LRFitNoResponses",
                 "[LinearRegressionFitMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("training", std::move(trainX));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that absence of training data is checked.
 */
TEST_CASE_METHOD(LRFitTestFixture, "LRFitNoTrainingData",
                 "[LinearRegressionFitMainTest][BindingTests]")
{
  constexpr int N = 10;

  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  SetInputParam("training_responses", std::move(trainY));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that error is thrown when negative regularization
 * is passed.
 */
TEST_CASE_METHOD(LRFitTestFixture, "LRFitNegRegularization",
                "[LinearRegressionFitMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);

  SetInputParam("training", std::move(trainX));
  SetInputParam("lambda", double(-1)); // negative regularization.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that the when no responses are given then last
 * row is automatically considered as responses.
 */
TEST_CASE_METHOD(LRFitTestFixture, "LRFitNoResponses2",
                "[LinearRegressionFitMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);

  SetInputParam("training", std::move(trainX));

  // Intentionally not passing training_responses.

  RUN_BINDING();
}
