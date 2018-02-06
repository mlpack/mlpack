/**
 * @file decision_stump_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct DecisionStumpTestFixture
{
 public:
  DecisionStumpTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~DecisionStumpTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(DecisionStumpMainTest, DecisionStumpTestFixture);

/**
 * Ensure that we get desired dimensions when both training
 * data and labels are passed.
 */
BOOST_AUTO_TEST_CASE(DecisionStumpOutputDimensionTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  // Delete the last row containing labels from input dataset.
  inputData.shed_row(inputData.n_rows - 1);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

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
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);

  // Check prediction have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
                      1);
}

/**
 * Check that last row of input file is used as labels
 * when labels are not passed specifically and results
 * are same from both label and labeless models.
 */
BOOST_AUTO_TEST_CASE(DecisionStumpLabelsLessDimensionTest)
{
  // Train DS without providing labels.
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Get the labels out.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", inputData);

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);

  // Check prediction have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
                      1);

  // Reset data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Store outputs.
  arma::Row<size_t> predictions;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

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
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);

  // Check prediction have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
                      1);

  // Check that initial output and final output matrix
  // from two models are same.
  CheckMatrices(predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(DecisionStumpModelReuseTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  arma::Row<size_t> predictions;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                std::move(CLI::GetParam<DSModel*>("output_model")));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);

  // Check predictions have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
                      1);

  // Check that initial predictions and final predicitons matrix
  // using saved model are same.
  CheckMatrices(predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
}

/**
 * Ensure that bucket_size is always positive.
 */
BOOST_AUTO_TEST_CASE(DecisionStumpBucketSizeTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("bucket_size", (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
BOOST_AUTO_TEST_CASE(DecisionStumpTrainingVerTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
                std::move(CLI::GetParam<DSModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
