/**
 * @file adaboost_test.cpp
 * @author Nikhil Goel
 *
 * Test mlpackMain() of adaboost_main.cpp.
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

  SetInputParam("test", std::move(testData));

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                std::move(CLI::GetParam<AdaBoostModel>("output_model")));

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
                std::move(CLI::GetParam<AdaBoostModel>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
 * Weak learner should be ignored if it is
 * specified with an input model file.
 */
BOOST_AUTO_TEST_CASE(AdaBoostWeakLearnerIgnoredTest)
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

  SetInputParam("test", std::move(testData));

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Default value is Decision Stump
  SetInputParam("input_model",
                std::move(CLI::GetParam<AdaBoostModel>("output_model")));
  SetInputParam("weak_learner", std::string("perceptron"));

  const string weakLearner = CLI::GetParam<string>("weak_learner");
  if (weakLearner == "perceptron")
  {
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
  }
}

BOOST_AUTO_TEST_SUITE_END();
