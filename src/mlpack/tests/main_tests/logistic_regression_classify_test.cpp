/**
  * @file logistic_regression_classify_test.cpp
  * @author B Kartheek Reddy
  *
  * Test RUN_BINDING() of logistic_regression_classify_main.cpp
  *
  * mlpack is free software; you may redistribute it and/or modify it under the
  * terms of the 3-clause BSD license.  You should have received a copy of the
  * 3-clause BSD license along with mlpack.  If not, see
  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
  */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_classify_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LogisticRegressionClassifyTestFixture);

/**
  * Ensuring that absence of test data is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionClassifyTestFixture,
                 "LogisticRegressionClassifyNoData",
                 "[LogisticRegressionClassifyMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;
  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D, M);

  // Training the model.
  LogisticRegression<>* model = new LogisticRegression<>(0, 0);
  //model->Parameters() = zeros<arma::rowvec>(trainX.n_rows + 1);
  //model->Train(trainX, trainY);

  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  // Test data is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
