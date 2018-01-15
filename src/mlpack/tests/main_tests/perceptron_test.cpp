/**
 * @file perceptron_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of perceptron_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "Perceptron";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/perceptron/perceptron_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct PerceptronTestFixture
{
 public:
  PerceptronTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PerceptronTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PerceptronMainTest,
                         PerceptronTestFixture);

/**
 * Ensure that we get desired dimensions when both training
 * data and labels are passed.
 */
BOOST_AUTO_TEST_CASE(PerceptronOutputDimensionTest)
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
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_cols,
                      testSize);

  // Check output have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_rows, 1);
}

/**
 * Check that last row of input file is used as labels
 * when labels are not passed specifically.
 */
BOOST_AUTO_TEST_CASE(PerceptronLabelsLessDimensionTest)
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
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_cols,
                      testSize);

  // Check output have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_rows, 1);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(PerceptronModelReuseTest)
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
  SetInputParam("test", std::move(testData));

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Cannot load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                std::move(CLI::GetParam<PerceptronModel>("output_model")));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_cols,
                      testSize);

  // Check output have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_rows, 1);

  // Check that initial output and final output matrix
  // using saved model are same.
  CheckMatrices(output, CLI::GetParam<arma::Row<size_t>>("output"));
}

/**
 * Ensure that max_iterations is always non-negative.
 */
BOOST_AUTO_TEST_CASE(PerceptronMaxItrTest)
{
  arma::mat inputData;
  if (!data::Load("trainSet.csv", inputData))
    BOOST_FAIL("Cannot load train dataset trainSet.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("max_iterations", (int) -1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
