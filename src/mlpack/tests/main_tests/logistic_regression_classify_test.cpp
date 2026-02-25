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
  * Ensuring that absence of input model is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionClassifyTestFixture,
                 "LogisticRegressionClassifyNoModel",
                 "[LogisticRegressionClassifyMainTest][BindingTests]")
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
TEST_CASE_METHOD(LogisticRegressionClassifyTestFixture,
                 "LogisticRegressionClassifyEmptyModel",
                 "[LogisticRegressionClassifyMainTest][BindingTests]")
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
TEST_CASE_METHOD(LogisticRegressionClassifyTestFixture,
                 "LogisticRegressionClassifyNoData",
                 "[LogisticRegressionClassifyMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  // Training the model.
  LogisticRegression<>* model = new LogisticRegression<>(trainX, trainY);

  SetInputParam("input_model", std::move(model));

  // Test data has the wrong dimensionality. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that wrong size of test data is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionClassifyTestFixture,
                 "LogisticRegressionClassifyWrongSizeData",
                 "[LogisticRegressionClassifyMainTest][BindingTests]")
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
TEST_CASE_METHOD(LogisticRegressionClassifyTestFixture,
                 "LogisticRegressionClassifyPredictionSize",
                 "[LogisticRegressionClassifyMainTest][BindingTests]")
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

  arma::Row<size_t> predictions = params.Get<arma::Row<size_t>>("predictions");

  // Check that number of predicted labels is equal to the input test points.
  REQUIRE(predictions.n_rows == 1);
  REQUIRE(predictions.n_cols == M);
}

/**
  * Ensuring decision_boundary parameter does something.
 **/
TEST_CASE_METHOD(LogisticRegressionClassifyTestFixture,
                 "LogisticRegressionClassifyDecisionBoundaryTest",
                 "[LogisticRegressionClassifyMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int M = 50;
  arma::mat trainX = {
    {0, 0, 0, 0, 1, 1, 1, 1, 22, 5},
    {0, 0, 1, 1, 0, 0, 1, 1, 33, 25},
    {0, 1, 0, 1, 0, 1, 0, 1, 44, 55}
  };
  arma::Row<size_t> trainY = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 1 };
  arma::mat testX = arma::randn<arma::mat>(D, M);

  // Training the model.
  LogisticRegression<>* model = new LogisticRegression<>(trainX, trainY);

  SetInputParam("input_model", model);
  SetInputParam("test", testX);
  SetInputParam("decision_boundary", double(0.1));

  // First solution.
  RUN_BINDING();

  // Get the output after first training.
  const arma::Row<size_t> output1 =
      params.Get<arma::Row<size_t>>("predictions");

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  // re-fit
  model = new LogisticRegression<>(0, 0);
  model->Parameters() = zeros<arma::rowvec>(D + 1);
  model->Train(trainX, trainY);

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));
  SetInputParam("decision_boundary", double(0.9));

  // Second solution.
  RUN_BINDING();

  // Get the output after second training.
  const arma::Row<size_t>& output2 =
      params.Get<arma::Row<size_t>>("predictions");

  // Check that the output changed when the decision boundary moved.
  REQUIRE(accu(output1 != output2) > 0);
}
