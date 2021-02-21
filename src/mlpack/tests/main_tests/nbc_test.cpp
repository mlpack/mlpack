/**
 * @file tests/main_tests/nbc_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of nbc_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "NBC";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/naive_bayes/nbc_main.cpp>
#include "test_helper.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

struct NBCTestFixture
{
 public:
  NBCTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~NBCTestFixture()
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
TEST_CASE_METHOD(NBCTestFixture, "NBCOutputDimensionTest",
                "[NBCMainTest][BindingTests]")
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
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_cols == testSize);

  // Check output have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_rows == 2);
}

/**
 * Check that last row of input file is used as labels
 * when labels are not passed specifically and results
 * are same from both label and labeless models.
 */
TEST_CASE_METHOD(NBCTestFixture, "NBCLabelsLessDimensionTest",
                "[NBCMainTest][BindingTests]")
{
  // Train NBC without providing labels.
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
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_cols == testSize);

  // Check output have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_rows == 2);

  // Reset data passed.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Store outputs.
  arma::Row<size_t> output;
  arma::mat output_probs;
  output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
  output_probs = std::move(IO::GetParam<arma::mat>("output_probs"));

  bindings::tests::CleanMemory();

  // Now train NBC with labels provided.

  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("test", std::move(testData));
  // Pass Labels.
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_cols == testSize);

  // Check output have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_rows == 2);

  // Check that initial output and final output matrix
  // from two models are same.
  CheckMatrices(output, IO::GetParam<arma::Row<size_t>>("output"));
  CheckMatrices(output_probs, IO::GetParam<arma::mat>("output_probs"));
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(NBCTestFixture, "NBCModelReuseTest",
                "[NBCMainTest][BindingTests]")
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

  arma::Row<size_t> output;
  arma::mat output_probs;
  output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
  output_probs = std::move(IO::GetParam<arma::mat>("output_probs"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                std::move(IO::GetParam<NBCModel*>("output_model")));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_cols == testSize);

  // Check output have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_rows == 2);

  // Check that initial output and final output
  // matrix using saved model are same.
  CheckMatrices(output, IO::GetParam<arma::Row<size_t>>("output"));
  CheckMatrices(output_probs, IO::GetParam<arma::mat>("output_probs"));
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
TEST_CASE_METHOD(NBCTestFixture, "NBCTrainingVerTest",
                "[NBCMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
                std::move(IO::GetParam<NBCModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that models trained with or without incremental
 * variance outputs same results
 */
TEST_CASE_METHOD(NBCTestFixture, "NBCIncrementalVarianceTest",
                "[NBCMainTest][BindingTests]")
{
  // Train NBC with incremental variance.
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
  SetInputParam("training", inputData);

  // Input test data.
  SetInputParam("test", testData);
  SetInputParam("incremental_variance", (bool) true);

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_cols == testSize);

  // Check output have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_rows == 2);

  bindings::tests::CleanMemory();

  // Reset data passed.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["incremental_variance"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Store outputs.
  arma::Row<size_t> output;
  arma::mat output_probs;
  output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
  output_probs = std::move(IO::GetParam<arma::mat>("output_probs"));

  // Now train NBC without incremental_variance.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("test", std::move(testData));
  SetInputParam("incremental_variance", (bool) false);

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_cols == testSize);

  // Check output have only single row.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("output_probs").n_rows == 2);

  // Check that initial output and final output matrix
  // from two models are same.
  CheckMatrices(output, IO::GetParam<arma::Row<size_t>>("output"));
  CheckMatrices(output_probs, IO::GetParam<arma::mat>("output_probs"));
}

/**
 * Ensure that the parameter 'output' and the parameter 'predictions' give the
 * same output.  This test case should be removed in mlpack 4 when the
 * deprecated parameter 'output' is removed.
 */
TEST_CASE_METHOD(NBCTestFixture, "NBCOptionConsistencyTest",
                "[NBCMainTest][BindingTests]")
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

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Get the output from the 'output' parameter.
  const arma::Row<size_t> testY1 =
      std::move(IO::GetParam<arma::Row<size_t>>("output"));

  // Get output from 'predictions' parameter.
  const arma::Row<size_t> testY2 =
      IO::GetParam<arma::Row<size_t>>("predictions");

  // Both solutions must be equal.
  CheckMatrices(testY1, testY2);
}


/**
 * This test ensures that the parameter 'output_probabilities' and the parameter
 * 'probabilities' give the same output.  This test case should be removed in
 * mlpack 4 when the deprecated parameter: 'output_probabilities' is removed.
 */
TEST_CASE_METHOD(NBCTestFixture, "NBCOptionConsistencyTest2",
                "[NBCMainTest][BindingTests]")
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

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Get the output probabilites which is a deprecated parameter.
  const arma::mat testY1 =
      std::move(IO::GetParam<arma::mat>("output_probs"));

  // Get probabilities from 'predictions' parameter.
  const arma::mat testY2 =
      IO::GetParam<arma::mat>("probabilities");

  // Both solutions must be equal.
  CheckMatrices(testY1, testY2);
}
