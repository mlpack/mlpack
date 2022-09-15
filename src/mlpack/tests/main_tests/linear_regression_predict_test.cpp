/**
 * @file tests/main_tests/linear_regression_predict_test.cpp
 * @author Nippun Sharma
 *
 * Test RUN_BINDING() of linear_regression_predict_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression_predict_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LRPredictTestFixture);

/**
 * Ensuring that test data dimensionality is checked.
 */
TEST_CASE_METHOD(LRPredictTestFixture, "LRPredictWrongDimOfDataTest1t",
                 "[LinearRegressionPredictMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D - 1, M); // Wrong dimensionality.

  LinearRegression* model = new LinearRegression();
  model->Train(trainX, trainY);

  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
TEST_CASE_METHOD(LRPredictTestFixture, "LRPredictPredictionSizeCheck",
                 "[LinearRegressionPredictMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D, M);

  LinearRegression* model = new LinearRegression();
  model->Train(trainX, trainY);

  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  RUN_BINDING();

  const arma::rowvec testY = params.Get<arma::rowvec>("output_predictions");

  REQUIRE(testY.n_rows == 1);
  REQUIRE(testY.n_cols == M);
}
