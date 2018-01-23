/**
 * @file hoeffding_tree_test.cpp
 * @author Haritha Nair
 *
 * Test mlpackMain() of hoeffding_tree_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "HoeffdingTree";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;
using namespace data;

struct HoeffdingTreeTestFixture
{
 public:
  HoeffdingTreeTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~HoeffdingTreeTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(HoeffdingTreeMainTest,
                         HoeffdingTreeTestFixture);

/**
 * Check that number of output points and
 * number of input points are equal.
 */
BOOST_AUTO_TEST_CASE(HoeffdingTreeOutputDimensionTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 3);
}

/**
 * Check that number of output points and number
 * of input points are equal for categorical dataset.
 */
BOOST_AUTO_TEST_CASE(HoeffdingTreeCategoricalOutputDimensionTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    BOOST_FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>
      ("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 6);
}

/**
 * Check whether providing labels explicitly and extracting from last
 * dimension give the same output.
 */
BOOST_AUTO_TEST_CASE(HoeffdingTreeLabelLessTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("tae.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset tae.csv!");

  // Extract the labels.
  arma::Row<size_t> labels(inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    labels[i] = inputData(inputData.n_rows - 1, i);

  arma::mat testData;
  if (!data::Load("tae_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset tae_test.csv!");

  // Remove labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));

  // Input test data.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 3);

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("test", std::move(std::make_tuple(info, testData)));
  // Pass Labels.
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_cols, testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 3);

  // Check that initial and current predictions are same.
  CheckMatrices(
      predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(
      probabilities, CLI::GetParam<arma::mat>("probabilities"));
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(HoeffdingModelReuseTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));

  mlpackMain();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input trained model.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));
  SetInputParam("input_model",
      std::move(CLI::GetParam<HoeffdingTreeModel>("output_model")));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_cols, testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 3);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(
      predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(
      probabilities, CLI::GetParam<arma::mat>("probabilities"));
}

/**
 * Ensure that saved model trained on categorical dataset can be used again.
 */
BOOST_AUTO_TEST_CASE(HoeffdingModelCategoricalReuseTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    BOOST_FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));

  mlpackMain();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  if (!data::Load("braziltourism_test.arff", testData, info))
    BOOST_FAIL("Cannot load test dataset braziltourism_test.arff!");

  // Input trained model.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));
  SetInputParam("input_model",
      std::move(CLI::GetParam<HoeffdingTreeModel>("output_model")));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_cols, testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 6);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(
      predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(
      probabilities, CLI::GetParam<arma::mat>("probabilities"));
}

BOOST_AUTO_TEST_SUITE_END();
