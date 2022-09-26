/**
 * @file tests/main_tests/softmax_regression_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of softmax_regression_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(SoftmaxRegressionTestFixture);

/**
 * Ensure that we get desired dimensions when both training
 * data and labels are passed.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionOutputDimensionTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(testData));

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);

  // Check prediction have only single row.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
}

/**
 * Ensure that labels are necessarily passed when training.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionLabelsLessDimensionTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionModelReuseTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", testData);

  RUN_BINDING();

  arma::Row<size_t> predictions;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));

  // Reset passed parameters.
  SoftmaxRegression* m = params.Get<SoftmaxRegression*>("output_model");
  params.Get<SoftmaxRegression*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model", m);

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);

  // Check predictions have only single row.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);

  // Check that initial predictions and final predicitons matrix
  // using saved model are same.
  CheckMatrices(predictions, params.Get<arma::Row<size_t>>("predictions"));
}

/**
 * Ensure that max_iterations is always non-negative.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionMaxItrTest", "[SoftmaxRegressionMainTest][BindingsTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("max_iterations", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that lambda is always non-negative.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionLambdaTest", "[SoftmaxRegressionMainTest][BindingsTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("lambda", (double) -0.1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that number_of_classes is always positive.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionNumClassesTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("number_of_classes", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionTrainingVerTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  // Input pre-trained model.
  SetInputParam("input_model",
                params.Get<SoftmaxRegression*>("output_model"));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that output object parameters are
 * different for different lambda values.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionDiffLambdaTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  // Train SR for lambda 0.1.
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("lambda", (double) 0.1);

  // Input test data.
  SetInputParam("test", testData);

  RUN_BINDING();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = params.Get<SoftmaxRegression*>("output_model")->Parameters();

  // Reset passed parameters.
  CleanMemory();
  ResetSettings();

  // Train SR for lamda 0.9.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("lambda", (double) 0.9);
  SetInputParam("test", std::move(testData));

  RUN_BINDING();

  // Check that initial parameters and final parameters matrix
  // using saved model are different.
  for (size_t i = 0; i < modelParam.n_elem; ++i)
  {
    REQUIRE(modelParam[i] !=
        params.Get<SoftmaxRegression*>("output_model")->Parameters()[i]);
  }
}

/**
 * Check that output object parameters are different for different numbers of
 * max_iterations.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionDiffMaxItrTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  // Train SR for lambda 0.1.
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("max_iterations", (int) 500);

  // Input test data.
  SetInputParam("test", testData);

  RUN_BINDING();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = params.Get<SoftmaxRegression*>("output_model")->Parameters();

  // Reset passed parameters.
  CleanMemory();
  ResetSettings();

  // Train SR for lamda 0.9.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("max_iterations", (int) 1000);
  SetInputParam("test", std::move(testData));

  RUN_BINDING();

  // Check that initial parameters and final parameters matrix
  // using saved model are different.
  for (size_t i = 0; i < modelParam.n_elem; ++i)
  {
    REQUIRE(modelParam[i] !=
        params.Get<SoftmaxRegression*>("output_model")->Parameters()[i]);
  }
}

/**
 * Check that output object parameter for no_intercept
 * term is one less than with intercept term.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionDiffInterceptTest",
    "[SoftmaxRegressionMainTest][BindingsTests]")
{
  // Train SR with intercept.
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("no_intercept", (bool) true);

  // Input test data.
  SetInputParam("test", testData);

  RUN_BINDING();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = params.Get<SoftmaxRegression*>("output_model")->Parameters();

  // Reset passed parameters.
  CleanMemory();
  ResetSettings();

  // Train SR for no_intercept.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("test", std::move(testData));

  RUN_BINDING();

  // Check that initial parameters has 1 more parameter than
  // final parameters matrix.
  REQUIRE(
      params.Get<SoftmaxRegression*>("output_model")->Parameters().n_cols ==
      modelParam.n_cols + 1);
}

/**
 * Check that we can get output probabilities, and also that they are
 * reasonable.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionProbabilitiesTest",
    "[SoftmaxRegressionMainTest][BindingsTest]")
{
  // Train softmax regression.
  arma::mat data;
  if (!data::Load("vc2.csv", data))
    FAIL("Cannot load train dataset 'vc2.csv'!");
  // Get the labels out.
  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load training labels 'vc2_labels.txt'!");

  // Input training data.
  SetInputParam("training", data);
  SetInputParam("labels", labels);
  SetInputParam("no_intercept", (bool) true);

  // Input test data.
  SetInputParam("test", data);

  RUN_BINDING();

  // Get predictions and probabilities.
  arma::Row<size_t>& predictions =
      params.Get<arma::Row<size_t>>("predictions");
  arma::mat& probabilities = params.Get<arma::mat>("probabilities");

  REQUIRE(predictions.n_elem == probabilities.n_cols);
  REQUIRE(probabilities.n_rows == arma::max(labels) + 1);

  // Manually compute the predictions and ensure they match, and also check that
  // the probabilities sum to 1.
  for (size_t i = 0; i < probabilities.n_cols; ++i)
  {
    const double sum = arma::accu(probabilities.col(i));
    REQUIRE(sum == Approx(1.0));

    size_t classPrediction = (size_t) arma::index_max(probabilities.col(i));
    REQUIRE(classPrediction == predictions[i]);
  }
}
