/**
 * @file tests/main_tests/softmax_regression_probabilities_test.cpp
 * @author Manish Kumar
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of softmax_regression_probabilities_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_probabilities_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(SoftmaxRegressionProbabilitiesTestFixture);

/**
  * Ensuring that absence of input model is checked.
 **/
TEST_CASE_METHOD(SoftmaxRegressionProbabilitiesTestFixture,
                 "SoftmaxRegressionProbabilitiesNoModel",
                 "[SoftmaxRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int M = 15;
  arma::mat testX = arma::randu<arma::mat>(D, M);
  SetInputParam("test", std::move(testX));
  // (Required) model is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that absence of estimated model is checked.
 **/
TEST_CASE_METHOD(SoftmaxRegressionProbabilitiesTestFixture,
                 "SoftmaxRegressionProbabilitiesEmptyModel",
                 "[SoftmaxRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int M = 15;
  arma::mat testX = arma::randu<arma::mat>(D, M);
  SoftmaxRegression<>* model = new SoftmaxRegression<>;
  // model not trained
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));
  // Model is not trained so has no parameters.
  // Should throw invalid argument error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/**
  * Ensuring that absence of test data is checked.
 **/
TEST_CASE_METHOD(SoftmaxRegressionProbabilitiesTestFixture,
                 "SoftmaxRegressionProbabilitiesNoData",
                 "[SoftmaxRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  // Training the model.
  SoftmaxRegression<>* model = new SoftmaxRegression<>;
  model->Train(trainX, trainY, 2);
  SetInputParam("input_model", std::move(model));
  // Test data is missing. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that wrong size of test data is checked.
 **/
TEST_CASE_METHOD(SoftmaxRegressionProbabilitiesTestFixture,
                 "SoftmaxRegressionProbabilitiesWrongSizeData",
                 "[SoftmaxRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;
  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D - 1, M);

  // Training the model.
  SoftmaxRegression<>* model = new SoftmaxRegression<>;
  model->Train(trainX, trainY, 2, 0.001);
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));
  // Input data is of wrong size. Should throw invalid argument error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/**
  * Ensuring that predictions have right size
 **/
TEST_CASE_METHOD(SoftmaxRegressionProbabilitiesTestFixture,
                 "SoftmaxRegressionProbabilitiesPredictionSize",
                 "[SoftmaxRegressionProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 7;
  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 2, 1, 1, 2, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D, M);
  // Training the model.
  SoftmaxRegression<>* model = new SoftmaxRegression<>;
  model->Train(trainX, trainY, 3);
  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));
  RUN_BINDING();

  arma::mat probs = params.Get<arma::mat>("probabilities");
  // Check that number of probabilities is equal to the number input test points
  // times the number of classes
  REQUIRE(probs.n_rows == 3);
  REQUIRE(probs.n_cols == M);
  // Also check that probabilities are sensible, sum to 1
  // and are bounded by [0,1]
  for (size_t i = 0; i < probs.n_cols; ++i)
  {
    const double sum = accu(probs.col(i));
    REQUIRE(sum == Approx(1.0));

    REQUIRE(min(probs.col(i)) >= 0.0);
    REQUIRE(max(probs.col(i)) <= 1.0);
  }
}
