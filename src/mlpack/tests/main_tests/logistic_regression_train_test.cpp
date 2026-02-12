/**
  * @file logistic_regression_train_test.cpp
  * @author B Kartheek Reddy
  *
  * Test RUN_BINDING() of logistic_regression_train_main.cpp
  *
  * mlpack is free software; you may redistribute it and/or modify it under the
  * terms of the 3-clause BSD license.  You should have received a copy of the
  * 3-clause BSD license along with mlpack.  If not, see
  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
  */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LogisticRegressionTrainTestFixture);

/**
  * Ensuring that absence of training data is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionTrainTestFixture,
                 "LogisticRegressionTrainNoTrainingData",
                 "[LogisticRegressionTrainMainTest][BindingTests]")
{
  arma::Row<size_t> trainY;
  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  SetInputParam("labels", std::move(trainY));

  // Training data is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
/**
 * Ensuring that absence of responses is checked.
 */
TEST_CASE_METHOD(LogisticRegressionTrainTestFixture,
                 "LogisticRegressionTrainNoResponses",
                 "[LogisticRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("training", std::move(trainX));

  // Labels to the training data is not provided. It should throw
  // a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that the response size is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionTrainTestFixture,
                 "LogisticRegressionTrainWrongResponseSizeTest",
                 "[LogisticRegressionTrainMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int N = 10;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY; // Response vector with wrong size.

  // 8 responses - incorrect size.
  trainY = { 0, 0, 1, 0, 1, 1, 1, 0 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Labels with incorrect size. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that training responses contain only two classes (0 or 1).
 **/
TEST_CASE_METHOD(LogisticRegressionTrainTestFixture,
                 "LogisticRegressionTrainTrainWithMoreThanTwoClasses",
                 "[LogisticRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 8;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 8 responses containing more than two classes.
  trainY = { 0, 1, 0, 1, 2, 1, 3, 1 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training data contains more than two classes. It should throw
  // a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that max iteration for optimizers is non negative.
 **/
TEST_CASE_METHOD(LogisticRegressionTrainTestFixture,
                 "LogisticRegressionTrainNonNegativeMaxIterationTest",
                 "[LogisticRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", int(-1));

  // Maximum iterations is negative. It should a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that tolerance is non negative.
 **/
TEST_CASE_METHOD(LogisticRegressionTrainTestFixture,
                 "LogisticRegressionTrainNonNegativeToleranceTest",
                 "[LogisticRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 1, 1, 0, 1, 0, 0, 0, 1, 0, 1 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("tolerance", double(-0.01));

  // Tolerance is negative. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
