/**
 * @file tests/main_tests/perceptron_train_test.cpp
 * @author Manish Kumar
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of perceptron_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(PerceptronTrainTestFixture);

/**
 * Ensure that training data are passed when training.
 */
TEST_CASE_METHOD(PerceptronTrainTestFixture,
                 "PerceptronTrainNoTrainingData",
                 "[PerceptronTrainMainTest][BindingTests]")
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
TEST_CASE_METHOD(PerceptronTrainTestFixture,
                 "PerceptronTrainOutputDimensionTest",
                 "[PerceptronTrainMainTest][BindingTests]")
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
  params.Get<PerceptronModel*>("output_model")->P().Classify(testX, preds);

  // Check that number of output points are equal to number of input points.
  REQUIRE(preds.n_cols == testSize);
  // Check output have only single row.
  REQUIRE(preds.n_rows == 1);
}

/**
 * Check that last row of input file is used as labels
 * when labels are not passed specifically and results
 * are same from both label and labeless models.
 */
TEST_CASE_METHOD(PerceptronTrainTestFixture,
                 "PerceptronTrainLabelsLessDimensionTest",
                 "[PerceptronTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  constexpr int M = 5;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D + 1));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));
  arma::mat testX = arma::trans(arma::randu<arma::mat>(M, D));
  size_t testSize = testX.n_cols;

  // Train perceptron without providing labels.
  trainX.row(D) = arma::conv_to<arma::rowvec>::from(trainY);

  // Input training data.
  SetInputParam("training", trainX);
  RUN_BINDING();

  arma::Row<size_t> preds1;
  params.Get<PerceptronModel*>("output_model")->P().Classify(testX, preds1);

  // Check that number of output points are equal to number of input points.
  REQUIRE(preds1.n_cols == testSize);

  // Check output have only single row.
  REQUIRE(preds1.n_rows == 1);

  trainX.shed_row(D);

  // Reset data passed.
  CleanMemory();
  ResetSettings();

  // Now train perceptron with labels provided.
  // Input training data.
  SetInputParam("training", std::move(trainX));
  // Pass Labels.
  SetInputParam("labels", std::move(trainY));
  RUN_BINDING();

  arma::Row<size_t> preds2;
  params.Get<PerceptronModel*>("output_model")->P().Classify(testX, preds2);

  // Check that number of output points are equal to number of input points.
  REQUIRE(preds2.n_cols == testSize);

  // Check output have only single row.
  REQUIRE(preds2.n_rows == 1);

  // Check that initial output and final output matrix
  // from two models are same.
  REQUIRE(arma::approx_equal(preds1, preds2, "absdiff", 1e-7));
}


/**
 * Ensure that max_iterations is always non-negative.
 */
TEST_CASE_METHOD(PerceptronTrainTestFixture,
                 "PerceptronTrainMaxItrTest",
                 "[PerceptronTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D + 1));

  // Input training data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("max_iterations", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that the response size is checked.
 **/
TEST_CASE_METHOD(PerceptronTrainTestFixture,
                 "PerceptronTrainWrongResponseSizeTest",
                 "[PerceptronTrainMainTest][BindingTests]")
{
  constexpr int D = 2;
  constexpr int N = 10;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY; // Response vector with wrong size.

  // 8 responses.
  trainY = { 0, 0, 1, 0, 1, 1, 1, 0 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Labels for training data have wrong size. It should give runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that absence of responses is checked.
 */
TEST_CASE_METHOD(PerceptronTrainTestFixture,
                 "PerceptronTrainNoResponsesTest",
                 "[PerceptronTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("training", std::move(trainX));

  // No labels for training data. It should give runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that absence of training data is checked.
 */
TEST_CASE_METHOD(PerceptronTrainTestFixture,
                 "PerceptronTrainNoTrainingDataTest",
                 "[PerceptronTrainMainTest][BindingTests]")
{
  arma::Row<size_t> trainY;
  trainY = { 1, 1, 0, 1, 0, 0 };

  SetInputParam("labels", std::move(trainY));

  // No training data. It should give runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
