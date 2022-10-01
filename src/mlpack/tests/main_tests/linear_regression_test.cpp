/**
 * @file tests/main_tests/linear_regression_test.cpp
 * @author Eugene Freyman
 *
 * Test RUN_BINDING() of linear_regression_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LRTestFixture);

/**
 * Training a model with different regularization parameter and ensuring that
 * predictions are different.
 */
TEST_CASE_METHOD(LRTestFixture, "LRDifferentLambdas",
                 "[LinearRegressionMainTest][BindingTests]")
{
  // A required minimal difference between solutions.
  const double delta = 0.1;

  arma::mat trainX({1.0, 2.0, 3.0});
  arma::mat testX({4.0});
  arma::rowvec trainY({1.0, 4.0, 9.0});

  SetInputParam("training", trainX);
  SetInputParam("training_responses", trainY);
  SetInputParam("test", testX);
  SetInputParam("lambda", 0.1);

  // The first solution.
  RUN_BINDING();
  const double testY1 = params.Get<arma::rowvec>("output_predictions")(0);

  ResetSettings();

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", std::move(testX));
  SetInputParam("lambda", 1.0);

  // The second solution.
  RUN_BINDING();
  const double testY2 = params.Get<arma::rowvec>("output_predictions")(0);

  // Second solution has stronger regularization,
  // so the predicted value should be smaller.
  REQUIRE(testY1 - delta > testY2);
}


/**
 * Checking two options of specifying responses (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
TEST_CASE_METHOD(LRTestFixture, "LRResponsesRepresentation",
                 "[LinearRegressionMainTest][BindingTests]")
{
  constexpr double delta = 1e-5;

  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}});
  arma::mat testX({4.0});
  SetInputParam("training", trainX1);
  SetInputParam("test", testX);

  // The first solution.
  RUN_BINDING();
  const double testY1 = params.Get<arma::rowvec>("output_predictions")(0);

  CleanMemory();
  ResetSettings();

  arma::mat trainX2({1.0, 2.0, 3.0});
  arma::rowvec trainY2({1.0, 4.0, 9.0});
  SetInputParam("training", std::move(trainX2));
  SetInputParam("training_responses", std::move(trainY2));
  SetInputParam("test", std::move(testX));

  // The second solution.
  RUN_BINDING();
  const double testY2 = params.Get<arma::rowvec>("output_predictions")(0);

  REQUIRE(fabs(testY1 - testY2) < delta);
}

/**
 * Check that model can saved / loaded and used. Ensuring that results are the
 * same.
 */
TEST_CASE_METHOD(LRTestFixture, "LRModelReload",
                 "[LinearRegressionMainTest][BindingTests]")
{
  constexpr double delta = 1e-5;
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D, N);

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", testX);

  RUN_BINDING();

  LinearRegression* model = params.Get<LinearRegression*>("output_model");
  const arma::rowvec testY1 = params.Get<arma::rowvec>("output_predictions");

  ResetSettings();

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  RUN_BINDING();

  const arma::rowvec testY2 = params.Get<arma::rowvec>("output_predictions");

  double norm = arma::norm(testY1 - testY2, 2);
  REQUIRE(norm < delta);
}

/**
 * Ensuring that response size is checked.
 */
TEST_CASE_METHOD(LRTestFixture, "LRWrongResponseSizeTest",
                 "[LinearRegressionMainTest][BindingTests]")
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
 * Ensuring that test data dimensionality is checked.
 */
TEST_CASE_METHOD(LRTestFixture, "LRWrongDimOfDataTest1t",
                 "[LinearRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D - 1, M); // Wrong dimensionality.

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", std::move(testX));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
TEST_CASE_METHOD(LRTestFixture, "LRWrongDimOfDataTest2",
                 "[LinearRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));

  RUN_BINDING();

  LinearRegression* model = params.Get<LinearRegression*>("output_model");

  ResetSettings();

  arma::mat testX = arma::randu<arma::mat>(D - 1, M); // Wrong dimensionality.
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
TEST_CASE_METHOD(LRTestFixture, "LRPredictionSizeCheck",
                 "[LinearRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", std::move(testX));

  RUN_BINDING();

  const arma::rowvec testY = params.Get<arma::rowvec>("output_predictions");

  REQUIRE(testY.n_rows == 1);
  REQUIRE(testY.n_cols == M);
}

/**
 * Ensuring that absence of responses is checked.
 */
TEST_CASE_METHOD(LRTestFixture, "LRNoResponses",
                 "[LinearRegressionMainTest][BindingTests]")
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
TEST_CASE_METHOD(LRTestFixture, "LRNoTrainingData",
                 "[LinearRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;

  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  SetInputParam("training_responses", std::move(trainY));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
