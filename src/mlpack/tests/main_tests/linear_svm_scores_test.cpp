/**
 * @file tests/main_tests/linear_svm_scores_test.cpp
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of logistic_regression_scores_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm/linear_svm_scores_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LinearSVMScoresTestFixture);

/**
  * Ensuring that absence of input model is checked.
 **/
TEST_CASE_METHOD(LinearSVMScoresTestFixture,
                 "LinearSVMScoresNoModel",
                 "[LinearSVMScoresMainTest][BindingTests]")
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
TEST_CASE_METHOD(LinearSVMScoresTestFixture,
                 "LinearSVMScoresEmptyModel",
                 "[LinearSVMScoresMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int M = 15;
  arma::mat testX = arma::randu<arma::mat>(D, M);
  LinearSVMModel* model = new LinearSVMModel;
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
TEST_CASE_METHOD(LinearSVMScoresTestFixture,
                 "LinearSVMScoresNoData",
                 "[LinearSVMScoresMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  // Training the model.
  LinearSVMModel* model = new LinearSVMModel;
  model->svm.Train(trainX, trainY, 2);
  SetInputParam("input_model", std::move(model));
  // Test data is absent. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that wrong size of test data is checked.
 **/
TEST_CASE_METHOD(LinearSVMScoresTestFixture,
                 "LinearSVMScoresWrongSizeData",
                 "[LinearSVMScoresMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;
  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D - 1, M);

  // Training the model.
  LinearSVMModel* model = new LinearSVMModel;
  model->svm.Train(trainX, trainY, 2, 0.001);
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));
  // Required input data is of wrong size. Should throw invalid argument error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/**
  * Ensuring that predictions have right size
 **/
TEST_CASE_METHOD(LinearSVMScoresTestFixture,
                 "LinearSVMScoresPredictionSize",
                 "[LinearSVMScoresMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 7;
  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D, M);
  // Training the model.
  LinearSVMModel* model = new LinearSVMModel;
  model->mappings = { 0, 1 };
  model->svm.Train(trainX, trainY, 2);
  auto v = model->mappings;
  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));
  RUN_BINDING();

  arma::mat scores = params.Get<arma::mat>("scores");
  REQUIRE(scores.n_rows == 2);
  REQUIRE(scores.n_cols == M);
}
