/**
 * @file tests/main_tests/bayesian_linear_regression_train_test.cpp
 * @author Clement Mercier
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of bayesian_linear_regression_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(BRTrainTestFixture);

/**
 * Check error thrown for missing input or response data, and model
 * returned on fit from complete inputs.
 */
TEST_CASE_METHOD(BRTrainTestFixture,
                 "BRTrainNoInputNoResponse",
                 "[BayesianLinearRegressionTrainMainTest][BindingTests]")
{
  uword n = 50, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y = omega * matX;

  BayesianLinearRegression<> model;

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("input", std::move(matX));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("responses", std::move(y));

  RUN_BINDING();

  BayesianLinearRegression<>* estimator =
      params.Get<BayesianLinearRegression<>*>("output_model");
  REQUIRE(estimator != nullptr);
}
