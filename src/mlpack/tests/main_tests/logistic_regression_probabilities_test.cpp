/**
  * @file logistic_regression_probabilities_test.cpp
  * @author B Kartheek Reddy
  *
  * Test RUN_BINDING() of logistic_regression_probabilities_main.cpp
  *
  * mlpack is free software; you may redistribute it and/or modify it under the
  * terms of the 3-clause BSD license.  You should have received a copy of the
  * 3-clause BSD license along with mlpack.  If not, see
  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
  */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_probabilities_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LogisticRegressionProbabilitiesTestFixture);

/**
  * Ensuring that absence of input model is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionProbabilitiesTestFixture,
                 "LogisticRegressionProbabilitiesNoModel",
                 "[LogisticRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("test", std::move(testX));

  // (Required) model is not provided. Should throw a runtime error.
  // NB this currently requires a patch applied here too
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that absence of estimated model is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionProbabilitiesTestFixture,
                 "LogisticRegressionProbabilitiesEmptyModel",
                 "[LogisticRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat testX = arma::randu<arma::mat>(D, M);

  LogisticRegression<>* model = new LogisticRegression<>(0, 0);
  // model not trained
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  // Model is not trained and has no parameters. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}


/**
  * Ensuring that absence of test data is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionProbabilitiesTestFixture,
                 "LogisticRegressionProbabilitiesNoData",
                 "[LogisticRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  // Training the model.
  LogisticRegression<>* model = new LogisticRegression<>(trainX, trainY);

  SetInputParam("input_model", std::move(model));

  // Required input data is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that wrong size of test data is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionProbabilitiesTestFixture,
                 "LogisticRegressionProbabilitiesWrongSizeData",
                 "[LogisticRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D - 1, M);

  // Training the model.
  LogisticRegression<>* model = new LogisticRegression<>(trainX, trainY);

  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  // Required input data is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that predictions have right size
 **/
TEST_CASE_METHOD(LogisticRegressionProbabilitiesTestFixture,
                 "LogisticRegressionProbabilitiesPredictionSize",
                 "[LogisticRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D, M);

  // Training the model.
  LogisticRegression<>* model = new LogisticRegression<>(trainX, trainY);

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  RUN_BINDING();

  arma::mat probabilities = params.Get<arma::mat>("probabilities");

  // Check that number of predicted labels is equal to the input test points.
  REQUIRE(probabilities.n_rows == 2);
  REQUIRE(probabilities.n_cols == M);
}
