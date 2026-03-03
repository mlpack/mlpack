/**
 * @file tests/main_tests/linear_regression_train_test.cpp
 * @author Nippun Sharma
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of linear_regression_fit_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LarsFitTestFixture);

/**
 * Ensuring a minimally viable fit works.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitMVPTest",
                 "[LarsTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(N, D);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  SetInputParam("input", std::move(trainX));
  SetInputParam("responses", std::move(trainY));
  RUN_BINDING();

  arma::rowvec preds;
  arma::mat testX = { 0.123, 0.134 };
  testX = trans(testX);
  params.Get<LARS<>*>("output_model")->Predict(testX, preds);
  REQUIRE(preds.n_elem == 1);
}

/**
 * Training a model with different regularization parameter lambda1 and
 * ensuring that predictions are different.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitDifferentLambda1s",
                 "[LarsTrainMainTest][BindingTests]")
{
  // A required minimal difference between solutions.
  const double delta = 0.1;

  arma::mat trainX({1.0, 2.0, 3.0});
  trainX = trans(trainX);

  arma::mat testX({4.0});
  arma::rowvec trainY({1.0, 4.0, 9.0});

  SetInputParam("input", trainX);
  SetInputParam("responses", trainY);
  SetInputParam("lambda1", 0.1);

  // The first solution.
  RUN_BINDING();
  arma::rowvec preds1;
  params.Get<LARS<>*>("output_model")->Predict(testX, preds1);
  const double testY1 = preds1(0);

  ResetSettings();

  SetInputParam("input", std::move(trainX));
  SetInputParam("responses", std::move(trainY));
  SetInputParam("lambda1", 0.9);

  // The second solution.
  RUN_BINDING();
  arma::rowvec preds2;
  params.Get<LARS<>*>("output_model")->Predict(testX, preds2);
  const double testY2 = preds2(0);

  REQUIRE(std::abs(testY1 - testY2) > delta);
}

/**
 * Training a model with different regularization parameter lambda2 and
 * ensuring that predictions are different.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitDifferentLambda2s",
                 "[LarsTrainMainTest][BindingTests]")
{
  // A required minimal difference between solutions.
  const double delta = 0.1;

  arma::mat trainX({1.0, 2.0, 3.0});
  trainX = trans(trainX);

  arma::mat testX({4.0});
  arma::rowvec trainY({1.0, 4.0, 9.0});

  SetInputParam("input", trainX);
  SetInputParam("responses", trainY);
  SetInputParam("lambda1", 0.1);
  SetInputParam("lambda2", 0.1);

  // The first solution.
  RUN_BINDING();
  arma::rowvec preds1;
  params.Get<LARS<>*>("output_model")->Predict(testX, preds1);
  const double testY1 = preds1(0);

  ResetSettings();

  SetInputParam("input", std::move(trainX));
  SetInputParam("responses", std::move(trainY));
  SetInputParam("lambda1", 0.9);
  SetInputParam("lambda2", 0.9);

  // The second solution.
  RUN_BINDING();
  arma::rowvec preds2;
  params.Get<LARS<>*>("output_model")->Predict(testX, preds2);
  const double testY2 = preds2(0);

  REQUIRE(std::abs(testY1 - testY2) > delta);
}


/**
 * Ensuring that absense of training data is checked.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitMissingTrainingDataTest",
                 "[LarsTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);

  // No required input data for training.
  SetInputParam("responses", std::move(trainY));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that response size is checked.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitWrongResponseSizeTest",
                 "[LarsTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N + 3); // Wrong size.

  SetInputParam("input", std::move(trainX));
  SetInputParam("responses", std::move(trainY));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that absence of responses is checked.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitNoResponses",
                 "[LarsTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("input", std::move(trainX));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that absence of training data is checked.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitNoTrainingData",
                 "[LarsTrainMainTest][BindingTests]")
{
  constexpr int N = 10;

  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  SetInputParam("responses", std::move(trainY));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that error is thrown when negative regularization
 * lambda1 is passed.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitNegL1Regularization",
                "[LarsTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);

  SetInputParam("input", std::move(trainX));
  SetInputParam("lambda1", double(-1)); // negative regularization.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that error is thrown when negative regularization
 * lambda2 is passed.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitNegL2Regularization",
                "[LarsTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);

  SetInputParam("input", std::move(trainX));
  SetInputParam("lambda2", double(-1)); // negative regularization.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Training a model with and without intercept should lead to intercept
 * estimates of non-zero and zero values, respectively.
 */
TEST_CASE_METHOD(LarsFitTestFixture, "LarsFitWithAndWithoutIntercept",
                 "[LarsTrainMainTest][BindingTests]")
{
  arma::mat trainX({1.0, 2.0, 3.0});
  trainX = trans(trainX);

  arma::mat testX({4.0});
  arma::rowvec trainY({1.0, 4.0, 9.0});

  SetInputParam("input", trainX);
  SetInputParam("responses", trainY);

  // The first solution.
  RUN_BINDING();
  double alpha1 = params.Get<LARS<>*>("output_model")->Intercept();
  REQUIRE(alpha1 != 0);

  ResetSettings();

  SetInputParam("input", std::move(trainX));
  SetInputParam("responses", std::move(trainY));
  SetInputParam("no_intercept", true);

  // The second solution.
  RUN_BINDING();
  double alpha2 = params.Get<LARS<>*>("output_model")->Intercept();
  CHECK(alpha2 == 0.0f);
}
