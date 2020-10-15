/**
 * @file tests/main_tests/softmax_regression_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of softmax_regression_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "SoftmaxRegression";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_main.cpp>
#include "test_helper.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

struct SoftmaxRegressionTestFixture
{
 public:
  SoftmaxRegressionTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~SoftmaxRegressionTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/**
 * Ensure that we get desired dimensions when both training
 * data and labels are passed.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionOutputDimensionTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
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

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);

  // Check prediction have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);
}

/**
 * Ensure that labels are necessarily passed when training.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionLabelsLessDimensionTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionModelReuseTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
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

  mlpackMain();

  arma::Row<size_t> predictions;
  predictions = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                IO::GetParam<SoftmaxRegression*>("output_model"));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);

  // Check predictions have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);

  // Check that initial predictions and final predicitons matrix
  // using saved model are same.
  CheckMatrices(predictions, IO::GetParam<arma::Row<size_t>>("predictions"));
}

/**
 * Ensure that max_iterations is always non-negative.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionMaxItrTest", "[SoftmaxRegressionMainTest][BindingTests]")
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that lambda is always non-negative.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionLambdaTest", "[SoftmaxRegressionMainTest][BindingTests]")
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that number_of_classes is always positive.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionNumClassesTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionTrainingVerTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
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

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
                IO::GetParam<SoftmaxRegression*>("output_model"));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that output object parameters are
 * different for different lambda values.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionDiffLambdaTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
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

  mlpackMain();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = IO::GetParam<SoftmaxRegression*>("output_model")->Parameters();

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Train SR for lamda 0.9.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("lambda", (double) 0.9);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial parameters and final parameters matrix
  // using saved model are different.
  for (size_t i = 0; i < modelParam.n_elem; ++i)
  {
    REQUIRE(modelParam[i] !=
        IO::GetParam<SoftmaxRegression*>("output_model")->Parameters()[i]);
  }
}

/**
 * Check that output object parameters are different for different numbers of
 * max_iterations.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionDiffMaxItrTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
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

  mlpackMain();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = IO::GetParam<SoftmaxRegression*>("output_model")->Parameters();

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Train SR for lamda 0.9.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("max_iterations", (int) 1000);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial parameters and final parameters matrix
  // using saved model are different.
  for (size_t i = 0; i < modelParam.n_elem; ++i)
  {
    REQUIRE(modelParam[i] !=
        IO::GetParam<SoftmaxRegression*>("output_model")->Parameters()[i]);
  }
}

/**
 * Check that output object parameter for no_intercept
 * term is one less than with intercept term.
 */
TEST_CASE_METHOD(
    SoftmaxRegressionTestFixture,
    "SoftmaxRegressionDiffInterceptTest",
    "[SoftmaxRegressionMainTest][BindingTests]")
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

  mlpackMain();

  // Store output parameters.
  arma::mat modelParam;
  modelParam = IO::GetParam<SoftmaxRegression*>("output_model")->Parameters();

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;
  IO::GetSingleton().Parameters()["no_intercept"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Train SR for no_intercept.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial parameters has 1 more parameter than
  // final parameters matrix.
  REQUIRE(
      IO::GetParam<SoftmaxRegression*>("output_model")->Parameters().n_cols ==
      modelParam.n_cols + 1);
}
