/**
 * @file tests/main_tests/bayesian_linear_regression_predict_test.cpp
 * @author Clement Mercier
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of bayesian_linear_regression_predict_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression_predict_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(BRPredictTestFixture);

/**
 * Check for correct dimensions on prediction returns, without 'stddevs'
 * and with it set.
 */
TEST_CASE_METHOD(BRPredictTestFixture,
                 "BRPredictResult",
                 "[BayesianLinearRegressionPredictMainTest][BindingTests]")
{
  uword n = 50, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y = omega * matX;

  BayesianLinearRegression<>* model =  new BayesianLinearRegression<>();
  model->Train(matX, y);

  SetInputParam("input_model", model);
  SetInputParam("test", matX);

  RUN_BINDING();

  REQUIRE(params.Get<arma::mat>("predictions").n_cols == n);
  REQUIRE(params.Get<arma::mat>("predictions").n_rows == 1);

  ResetSettings();

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(matX));
  SetInputParam("stddevs", true);

  RUN_BINDING();

  REQUIRE(params.Get<arma::mat>("predictions").n_cols == n);
  REQUIRE(params.Get<arma::mat>("predictions").n_rows == 2);
}
