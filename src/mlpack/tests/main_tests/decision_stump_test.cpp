/**
 * @file tests/main_tests/decision_stump_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of decision_stump_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "DecisionStump";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/decision_stump/decision_stump_main.cpp>
#include "test_helper.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

struct DecisionStumpTestFixture
{
 public:
  DecisionStumpTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~DecisionStumpTestFixture()
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
TEST_CASE_METHOD(DecisionStumpTestFixture, "DecisionStumpOutputDimensionTest",
                 "[DecisionStumpMainTest][BindingTests]")
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
 * Check that last row of input file is used as labels
 * when labels are not passed specifically and results
 * are same from both label and labeless models.
 */
TEST_CASE_METHOD(DecisionStumpTestFixture,
                 "DecisionStumpLabelsLessDimensionTest",
                 "[DecisionStumpMainTest][BindingTests]")
{
  // Train DS without providing labels.
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", inputData);

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);

  // Check prediction have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);

  // Reset data passed.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Store outputs.
  arma::Row<size_t> predictions;
  predictions = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

  // Delete the previous model.
  bindings::tests::CleanMemory();

  // Now train DS with labels provided.

  // Delete last row of inputData.
  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("test", std::move(testData));
  // Pass Labels.
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);

  // Check prediction have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);

  // Check that initial output and final output matrix
  // from two models are same.
  CheckMatrices(predictions, IO::GetParam<arma::Row<size_t>>("predictions"));
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(DecisionStumpTestFixture, "DecisionStumpModelReuseTest",
                 "[DecisionStumpMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  arma::Row<size_t> predictions;
  predictions = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                std::move(IO::GetParam<DSModel*>("output_model")));

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
 * Ensure that bucket_size is always positive.
 */
TEST_CASE_METHOD(DecisionStumpTestFixture, "DecisionStumpBucketSizeTest",
                 "[DecisionStumpMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("bucket_size", (int) 0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
TEST_CASE_METHOD(DecisionStumpTestFixture, "DecisionStumpTrainingVerTest",
                 "[DecisionStumpMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
                std::move(IO::GetParam<DSModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
