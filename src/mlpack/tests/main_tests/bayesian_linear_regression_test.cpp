/**
 * @file tests/main_tests/bayesian_linear_regression_test.cpp
 * @author Clement Mercier
 *
 * Test RUN_BINDING() of bayesian_linear_regression_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(BRTestFixture);

/**
 * Check the center and scale options.
 */
TEST_CASE_METHOD(BRTestFixture,
                 "BRCenter0Scale0",
                 "[BayesianLinearRegressionMainTest][BindingTests]")
{
  int n = 50, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y =  omega * matX;

  SetInputParam("input", std::move(matX));
  SetInputParam("responses", std::move(y));
  SetInputParam("center", false);

  RUN_BINDING();

  BayesianLinearRegression* estimator =
      params.Get<BayesianLinearRegression*>("output_model");

  REQUIRE(estimator->DataOffset().n_elem == 0);
  REQUIRE(estimator->DataScale().n_elem == 0);
}

/**
 * Check predictions of saved model and in code model are equal.
 */
TEST_CASE_METHOD(BRTestFixture,
                 "BayesianLinearRegressionSavedEqualCode",
                 "[BayesianLinearRegressionMainTest][BindingTests]")
{
  int n = 10, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::mat matXtest = arma::randu<arma::mat>(m, 2 * n);
  const arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y =  omega * matX;

  BayesianLinearRegression model;
  model.Train(matX, y);

  arma::rowvec responses;
  model.Predict(matXtest, responses);

  SetInputParam("input", std::move(matX));
  SetInputParam("responses", std::move(y));

  RUN_BINDING();

  BayesianLinearRegression* mOut =
      params.Get<BayesianLinearRegression*>("output_model");

  ResetSettings();

  SetInputParam("input_model", mOut);
  SetInputParam("test", std::move(matXtest));

  RUN_BINDING();

  arma::mat ytest = std::move(responses);
  // Check that initial output and output using saved model are same.
  CheckMatrices(ytest, params.Get<arma::mat>("predictions"));
}

/**
 * Check a crash happens if neither input or input_model are specified.
 * Check a crash happens if both input and input_model are specified.
 */
TEST_CASE_METHOD(BRTestFixture,
                 "CheckParamsPassed",
                 "[BayesianLinearRegressionMainTest][BindingTests]")
{
  int n = 10, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::mat matXtest = arma::randu<arma::mat>(m, 2 * n);
  const arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y =  omega * matX;

  BayesianLinearRegression model;
  model.Train(matX, y);

  arma::rowvec responses;
  model.Predict(matXtest, responses);

  // Check that std::runtime_error is thrown if neither input or input_model
  // is specified.
  SetInputParam("responses", std::move(y));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Continue only with input passed.
  SetInputParam("input", std::move(matX));
  RUN_BINDING();

  // Now pass the previous trained model and one input matrix at the same time.
  // An error should occur.
  SetInputParam("input", std::move(matX));
  SetInputParam("input_model",
                params.Get<BayesianLinearRegression*>("output_model"));
  SetInputParam("test", std::move(matXtest));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
