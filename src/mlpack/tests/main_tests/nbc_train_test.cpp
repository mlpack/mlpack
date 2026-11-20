/**
 * @file tests/main_tests/nbc_train_test.cpp
 * @author Manish Kumar
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of nbc_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/naive_bayes/nbc_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(NBCTrainTestFixture);

/**
 * Ensure that training data are passed when training.
 */
TEST_CASE_METHOD(NBCTrainTestFixture, "NBCTrainNoTrainingData",
                 "[NBCTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  arma::Row<size_t> trainLabels = arma::randu<arma::Row<size_t>>(N);
  SetInputParam("labels", std::move(trainLabels));

  // Training data is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that we get desired dimensions when both training
 * data and labels are passed.
 */
TEST_CASE_METHOD(NBCTrainTestFixture, "NBCTrainOutputDimensionTest",
                "[NBCTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  constexpr int M = 5;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));
  arma::mat testX = arma::trans(arma::randu<arma::mat>(M, D));
  size_t testSize = testX.n_cols;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training the model.
  RUN_BINDING();

  // Get the output predictions of the test data.
  arma::Row<size_t> preds;
  params.Get<NBCModel*>("output_model")->nbc.Classify(testX, preds);

  // Output predictions size must match the test data set size.
  REQUIRE(preds.n_rows == 1);
  REQUIRE(preds.n_cols == testSize);
}

/**
 * Check that last row of input file is used as labels
 * when labels are not passed and results are identical
 */
TEST_CASE_METHOD(NBCTrainTestFixture, "NBCTrainLabelsLessDimensionTest",
                "[NBCTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 5;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));
  // Set the labels.
  trainX.row(D - 1) = arma::conv_to<arma::rowvec>::from(trainY);
  arma::mat testX = arma::trans(arma::randu<arma::mat>(M, D - 1));

  // Train NBC without providing labels.
  // Input training data, then train.
  SetInputParam("training", trainX);
  RUN_BINDING();

  // Get the output predictions of the test data.
  arma::Row<size_t> preds1;
  params.Get<NBCModel*>("output_model")->nbc.Classify(testX, preds1);
  // Check that number of output points are equal to number of input points.
  REQUIRE(preds1.n_cols == M);
  REQUIRE(preds1.n_rows == 1);

  // Reset data passed.
  CleanMemory();
  ResetSettings();

  // Now train NBC with labels provided.
  trainX.shed_row(D - 1);
  // Input training data and labels, then train.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  RUN_BINDING();

  // Get the output predictions of the test data.
  arma::Row<size_t> preds2;
  params.Get<NBCModel*>("output_model")->nbc.Classify(testX, preds2);
  // Check that number of output points are equal to number of input points.
  REQUIRE(preds2.n_cols == M);
  REQUIRE(preds2.n_rows == 1);

  // Check that initial output and final output matrix
  // from two models are same.
  REQUIRE(arma::approx_equal(preds1, preds2, "absdiff", 1e-7));
}

/**
 * Check that models trained with or without incremental
 * variance outputs same results
 */
TEST_CASE_METHOD(NBCTrainTestFixture, "NBCTrainIncrementalVarianceTest",
                "[NBCTrainMainTest][BindingTests]")
{
  // Train NBC with incremental variance.
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 5;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));
  arma::mat testX = arma::trans(arma::randu<arma::mat>(M, D));

  // Input training data and labels. Set incremental variance.
  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("incremental_variance", (bool) true);
  RUN_BINDING();

  // Get the output predictions of the test data.
  arma::Row<size_t> preds1;
  params.Get<NBCModel*>("output_model")->nbc.Classify(testX, preds1);

  CleanMemory();
  ResetSettings();

  // Now train NBC without incremental_variance.
  // Input training and labels data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", trainY);
  SetInputParam("incremental_variance", (bool) false);
  RUN_BINDING();

  arma::Row<size_t> preds2;
  params.Get<NBCModel*>("output_model")->nbc.Classify(testX, preds2);

  // Check that initial output and final output matrix
  // from two models are same.
  REQUIRE(arma::approx_equal(preds1, preds2, "absdiff", 1e-7));
}
