/**
 * @file tests/main_tests/lars_predict_test.cpp
 * @author Nippun Sharma
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of lars_predict_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars_predict_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LarsPredictTestFixture);

/**
  * Ensuring that absence of input model is checked.
 **/
TEST_CASE_METHOD(LarsPredictTestFixture, "LarsPredictNoModel",
                 "[LarsPredictMainTest][BindingTests]")
{
  // (Required) input_model is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that test data size is checked
 **/
TEST_CASE_METHOD(LarsPredictTestFixture, "LarsPredictDataDim"
                 "[LarsPredictMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  constexpr int M = 5;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);

  // Training the model.
  LARS<>* model = new LARS<>(trainX, trainY);

  SetInputParam("input_model", std::move(model));

  // Test data with incorrect dims
  arma::mat testX = arma::randu<arma::mat>(M, D - 1);
  SetInputParam("test", std::move(testX));

  // No test data. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that predicting single data point works
 **/
TEST_CASE_METHOD(LarsPredictTestFixture, "LarsPredictionSinglePoint",
                 "[LarsPredictMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);

  // Training the model.
  LARS<>* model = new LARS<>(trainX, trainY);
  SetInputParam("input_model", std::move(model));

  // Test data point
  arma::mat testP = arma::randu<arma::mat>(1, D);
  SetInputParam("test", std::move(testP));
  RUN_BINDING();

  const arma::mat prediction = params.Get<arma::mat>("predictions");
  REQUIRE(prediction(0, 0) != 0);
  REQUIRE(prediction.n_elem == 1);
}

/**
  * Ensuring that predicting data points works
 **/
TEST_CASE_METHOD(LarsPredictTestFixture, "LarsPredictionPoints",
                 "[LarsPredictMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  constexpr int M = 5;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);

  // Training the model.
  LARS<>* model = new LARS<>(trainX, trainY);
  SetInputParam("input_model", std::move(model));

  // Test data points
  arma::mat testP = arma::randu<arma::mat>(M, D);
  SetInputParam("test", std::move(testP));
  RUN_BINDING();

  const arma::mat prediction = params.Get<arma::mat>("predictions");
  REQUIRE(arma::all(arma::all(prediction != 0) == 1) == 1);
  REQUIRE(prediction.n_elem == M);
}
