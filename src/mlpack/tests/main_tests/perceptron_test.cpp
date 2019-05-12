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
    bindings::tests::CleanMemory();
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
 * when labels are not passed specifically and results
 * are same from both label and labeless models.
 */
BOOST_AUTO_TEST_CASE(PerceptronLabelsLessDimensionTest)
{
  // Train perceptron without providing labels.
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
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_cols,
                      testSize);

  // Check output have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_rows, 1);

  // Reset data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  inputData.shed_row(inputData.n_rows - 1);

  // Store outputs.
  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  bindings::tests::CleanMemory();

  // Now train perceptron with labels provided.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("test", std::move(testData));
  // Pass Labels.
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_cols,
                      testSize);

  // Check output have only single row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_rows, 1);

  // Check that initial output and final output matrix
  // from two models are same.
  CheckMatrices(output, CLI::GetParam<arma::Row<size_t>>("output"));
}

/**
 * This test can be removed in mlpack 4.0.0. This tests that the output and
 * predictions outputs are the same.
 */
BOOST_AUTO_TEST_CASE(PerceptronOutputPredictionsCheck)
{
  arma::mat trainX1;
  arma::Row<size_t> labelsX1;

  // Loading a train data set with 3 classes.
  if (!data::Load("vc2.csv", trainX1))
  {
    BOOST_FAIL("Could not load the train data (vc2.csv)");
  }

  // Loading the corresponding labels to the dataset.
  if (!data::Load("vc2_labels.txt", labelsX1))
  {
    BOOST_FAIL("Could not load the train data (vc2_labels.csv)");
  }

  SetInputParam("training", std::move(trainX1)); // Training data.
  // Labels for the training data.
  SetInputParam("labels", std::move(labelsX1));

  // Training model using first training dataset.
  mlpackMain();

  // Check that the outputs are the same.
  CheckMatrices(CLI::GetParam<arma::Row<size_t>>("output"),
                CLI::GetParam<arma::Row<size_t>>("predictions"));
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
  SetInputParam("test", testData);

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                CLI::GetParam<PerceptronModel*>("output_model"));

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

/**
  * Ensuring that re-training of an existing model
  * with different of classes is checked.
 **/
BOOST_AUTO_TEST_CASE(PerceptronReTrainWithWrongClasses)
{
  arma::mat trainX1;
  arma::Row<size_t> labelsX1;

  // Loading a train data set with 3 classes.
  if (!data::Load("vc2.csv", trainX1))
  {
    BOOST_FAIL("Could not load the train data (vc2.csv)");
  }

  // Loading the corresponding labels to the dataset.
  if (!data::Load("vc2_labels.txt", labelsX1))
  {
    BOOST_FAIL("Could not load the train data (vc2_labels.csv)");
  }

  SetInputParam("training", std::move(trainX1)); // Training data.
  // Labels for the training data.
  SetInputParam("labels", std::move(labelsX1));

  // Training model using first training dataset.
  mlpackMain();

  // Get the output model obtained after training.
  PerceptronModel* model =
      CLI::GetParam<PerceptronModel*>("output_model");

  // Reset the data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;

  // Creating training data with five classes.
  constexpr int D = 3;
  constexpr int N = 10;
  arma::mat trainX2 = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> labelsX2;

  // 10 responses.
  labelsX2 << 0 << 1 << 4 << 1 << 2 << 1 << 0 << 3 << 3 << 0 << endr;

  // Last column of trainX2 contains the class labels.
  SetInputParam("training", std::move(trainX2));
  SetInputParam("input_model", model);

  // Re-training an existing model of 3 classes
  // with training data of 5 classes. It should give runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Checking for dimensionality of the test data set.
 **/
BOOST_AUTO_TEST_CASE(PerceptronWrongDimOfTestData)
{
  constexpr int N = 10;
  constexpr int D = 4;
  constexpr int M = 20;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << endr;

  // Test data with wrong dimensionality.
  arma::mat testX = arma::randu<arma::mat>(D-3, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", std::move(testX));

  // Test data set with wrong dimensionality. It should give runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring that the response size is checked.
 **/
BOOST_AUTO_TEST_CASE(PerceptronWrongResponseSizeTest)
{
  constexpr int D = 2;
  constexpr int N = 10;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY; // Response vector with wrong size.

  // 8 responses.
  trainY << 0 << 0 << 1 << 0 << 1 << 1 << 1 << 0 << endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Labels for training data have wrong size. It should give runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that absence of responses is checked.
 */
BOOST_AUTO_TEST_CASE(PerceptronNoResponsesTest)
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("training", std::move(trainX));

  // No labels for training data. It should give runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that absence of training data is checked.
 */
BOOST_AUTO_TEST_CASE(PerceptronNoTrainingDataTest)
{
  arma::Row<size_t> trainY;
  trainY << 1 << 1 << 0 << 1 << 0 << 0 <<endr;

  SetInputParam("labels", std::move(trainY));

  // No training data. It should give runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
BOOST_AUTO_TEST_CASE(PerceptronWrongDimOfTestData2)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training the model.
  mlpackMain();

  // Get the output model obtained after the training.
  PerceptronModel* model =
      CLI::GetParam<PerceptronModel*>("output_model");

  // Reset the data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;

  // Test data with Wrong dimensionality.
  arma::mat testX = arma::randu<arma::mat>(D - 1, M);
  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  // Wrong dimensionality of test data. It should give runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
