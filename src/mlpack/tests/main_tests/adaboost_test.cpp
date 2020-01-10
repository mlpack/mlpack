/**
 * @file adaboost_test.cpp
 * @author Nikhil Goel
 *
 * Test mlpackMain() of adaboost_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "AdaBoost";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/adaboost/adaboost_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct AdaBoostTestFixture
{
 public:
  AdaBoostTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~AdaBoostTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(AdaBoostMainTest, AdaBoostTestFixture);

/**
 * Check that number of output labels and number of input
 * points are equal.
 */
BOOST_AUTO_TEST_CASE(AdaBoostOutputDimensionTest)
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Unable to load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of predicted labels is equal to the input test points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_cols,
                      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("output").n_rows, 1);
}

/**
 * Check that total number of rows of probabilities matrix is equal to total
 * number of rows of input data and that each column of probabilities matrix sums
 * up to 1.
 */
BOOST_AUTO_TEST_CASE(AdaBoostProbabilitiesTest)
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Unable to load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  SetInputParam("test", std::move(testData));

  mlpackMain();

  arma::mat probabilities;
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  BOOST_REQUIRE_EQUAL(probabilities.n_cols, testSize);

  for (size_t i = 0; i < testSize; i++)
    BOOST_REQUIRE_CLOSE(arma::accu(probabilities.col(i)), 1, 1e-5);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(AdaBoostModelReuseTest)
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Unable to load test dataset vc2.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  SetInputParam("test", testData);

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                CLI::GetParam<AdaBoostModel*>("output_model"));

  mlpackMain();

  // Check that initial output and output using saved model are same.
  CheckMatrices(output, CLI::GetParam<arma::Row<size_t>>("output"));
}

/**
 * Test that iterations in adaboost is always non-negative.
 */
BOOST_AUTO_TEST_CASE(AdaBoostItrTest)
{
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    BOOST_FAIL("Unable load train dataset trainSet.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("iterations", (int) -1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that the last dimension of the training set is
 * used as labels when labels are not passed specifically 
 * and results are same from both label and without label models.
 */
BOOST_AUTO_TEST_CASE(AdaBoostWithoutLabelTest)
{
  // Train adaboost without providing labels.
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    BOOST_FAIL("Unable to load train dataset trainSet.csv!");

  // Give labels.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    BOOST_FAIL("Unable to load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  SetInputParam("training", trainData);

  SetInputParam("test", testData);

  mlpackMain();

  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  bindings::tests::CleanMemory();

  trainData.shed_row(trainData.n_rows - 1);

  // Now train Adaboost with labels provided.
  SetInputParam("training", std::move(trainData));
  SetInputParam("test", std::move(testData));
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that initial output and final output matrix are same.
  CheckMatrices(output, CLI::GetParam<arma::Row<size_t>>("output"));
}

/**
 * Testing that only one of training data or pre-trained model is passed.
 */
BOOST_AUTO_TEST_CASE(AdaBoostTrainingDataOrModelTest)
{
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    BOOST_FAIL("Unable to load train dataset trainSet.csv!");

  SetInputParam("training", std::move(trainData));

  mlpackMain();

  SetInputParam("input_model",
                CLI::GetParam<AdaBoostModel*>("output_model"));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * This test can be removed in mlpack 4.0.0.  This tests that the output and
 * predictions outputs are the same.
 */
BOOST_AUTO_TEST_CASE(AdaBoostOutputPredictionsTest)
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  CheckMatrices(CLI::GetParam<arma::Row<size_t>>("output"),
                CLI::GetParam<arma::Row<size_t>>("predictions"));
}

/**
 * Weak learner should be either Decision Stump or Perceptron.
 */
BOOST_AUTO_TEST_CASE(AdaBoostWeakLearnerTest)
{
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    BOOST_FAIL("Unable to load train dataset trainSet.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("weak_learner", std::string("decision tree"));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Different Weak learner should give different outputs.
 */
BOOST_AUTO_TEST_CASE(AdaBoostDiffWeakLearnerOutputTest)
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Unable to load test dataset vc2.csv!");

  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("test", testData);

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("test", testData);
  SetInputParam("weak_learner", std::string("perceptron"));

  mlpackMain();

  arma::Row<size_t> outputPerceptron;
  outputPerceptron = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  BOOST_REQUIRE_GT(arma::accu(output != outputPerceptron), 1);
}

/**
 * Accuracy increases as Number of Iterations increases.
 * (Or converges and remains same)
 */
BOOST_AUTO_TEST_CASE(AdaBoostDiffItrTest)
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Unable to load test dataset vc2.csv!");

  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    BOOST_FAIL("Unable to load labels for vc2__test_labels.txt");

  // Iterations = 1
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("weak_learner", std::string("perceptron"));
  SetInputParam("iterations", (int) 1);

  mlpackMain();

  // Calculate accuracy.
  arma::Row<size_t> output;
  CLI::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  size_t correct = arma::accu(output == testLabels);
  double accuracy1 = (double(correct) / double(testLabels.n_elem) * 100);

  bindings::tests::CleanMemory();

  // Iterations = 10
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("weak_learner", std::string("perceptron"));
  SetInputParam("iterations", (int) 10);

  mlpackMain();

  // Calculate accuracy.
  CLI::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy10 = (double(correct) / double(testLabels.n_elem) * 100);

  bindings::tests::CleanMemory();

  // Iterations = 100
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("weak_learner", std::string("perceptron"));
  SetInputParam("iterations", (int) 100);

  mlpackMain();

  // Calculate accuracy.
  CLI::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy100 = (double(correct) / double(testLabels.n_elem) * 100);

  BOOST_REQUIRE_LE(accuracy1, accuracy10);
  BOOST_REQUIRE_LE(accuracy10, accuracy100);
}

/**
 * Accuracy increases as tolerance decreases.
 * (Execution Time also increases)
 */
BOOST_AUTO_TEST_CASE(AdaBoostDiffTolTest)
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Unable to load test dataset vc2.csv!");

  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    BOOST_FAIL("Unable to load labels for vc2__test_labels.txt");

  // tolerance = 0.001
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("tolerance", (double) 0.001);

  mlpackMain();

  // Calculate accuracy.
  arma::Row<size_t> output;
  CLI::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  size_t correct = arma::accu(output == testLabels);
  double accuracy1 = (double(correct) / double(testLabels.n_elem) * 100);

  bindings::tests::CleanMemory();

  // tolerance = 0.01
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("tolerance", (double) 0.01);

  mlpackMain();

  // Calculate accuracy.
  CLI::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy2 = (double(correct) / double(testLabels.n_elem) * 100);

  bindings::tests::CleanMemory();

  // tolerance = 0.1
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("tolerance", (double) 0.1);

  mlpackMain();

  // Calculate accuracy.
  CLI::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy3 = (double(correct) / double(testLabels.n_elem) * 100);

  BOOST_REQUIRE_LE(accuracy1, accuracy2);
  BOOST_REQUIRE_LE(accuracy2, accuracy3);
}

BOOST_AUTO_TEST_SUITE_END();
