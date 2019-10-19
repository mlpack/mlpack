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
    bindings::tests::CleanMemory();
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
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 1);
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
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>
      ("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 1);
}

/**
 * Check whether providing labels explicitly and extracting from last
 * dimension give the same output.
 */
BOOST_AUTO_TEST_CASE(HoeffdingTreeLabelLessTest)
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

  // Append labels to the training set.
  inputData.resize(inputData.n_rows+1, inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    inputData(inputData.n_rows-1, i) = labels[i];

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

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
      CLI::GetParam<arma::mat>("probabilities").n_rows, 1);

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  bindings::tests::CleanMemory();

  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("test", std::make_tuple(info, testData));
  // Pass Labels.
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_cols, testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 1);

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
      CLI::GetParam<HoeffdingTreeModel*>("output_model"));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_cols, testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 1);

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
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  if (!data::Load("braziltourism_test.arff", testData, info))
    BOOST_FAIL("Cannot load test dataset braziltourism_test.arff!");

  // Input trained model.
  SetInputParam("test", std::move(std::make_tuple(info, testData)));
  SetInputParam("input_model",
      CLI::GetParam<HoeffdingTreeModel*>("output_model"));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_cols, testSize);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_cols, testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::mat>("probabilities").n_rows, 1);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(
      predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(
      probabilities, CLI::GetParam<arma::mat>("probabilities"));
}

/**
 * Ensure that small min_samples creates larger model.
 */
BOOST_AUTO_TEST_CASE(HoeffdingMinSamplesTest)
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("min_samples", 10);
  SetInputParam("confidence", 0.25);

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;
  CLI::GetSingleton().Parameters()["min_samples"].wasPassed = false;
  CLI::GetSingleton().Parameters()["confidence"].wasPassed = false;

  nodes = (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes();

  bindings::tests::CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("min_samples", 2000);
  SetInputParam("confidence", 0.25);

  mlpackMain();

  // Check that small min_samples creates larger model.
  BOOST_REQUIRE_LT(
      (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes(),
      nodes);
}

/**
 * Ensure that large max_samples creates smaller model.
 */
BOOST_AUTO_TEST_CASE(HoeffdingMaxSamplesTest)
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("max_samples", 50000);
  SetInputParam("confidence", 0.95);

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;
  CLI::GetSingleton().Parameters()["max_samples"].wasPassed = false;
  CLI::GetSingleton().Parameters()["confidence"].wasPassed = false;

  nodes = (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes();

  bindings::tests::CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("max_samples", 5);
  SetInputParam("confidence", 0.95);

  mlpackMain();

  // Check that large max_samples creates smaller model.
  BOOST_REQUIRE_LT(nodes,
      (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes());
}

/**
 * Ensure that small confidence value creates larger model.
 */
BOOST_AUTO_TEST_CASE(HoeffdingConfidenceTest)
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("confidence", 0.95);

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;
  CLI::GetSingleton().Parameters()["confidence"].wasPassed = false;

  // Model with high confidence.
  nodes = (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes();

  bindings::tests::CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  // Model with low confidence.
  SetInputParam("confidence", 0.25);

  mlpackMain();
  // Check that higher confidence creates smaller tree.
  BOOST_REQUIRE_LT(nodes,
      (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes());
}

/**
 * Ensure that large number of passes creates larger model.
 */
BOOST_AUTO_TEST_CASE(HoeffdingPassesTest)
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("passes", 1);

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;
  CLI::GetSingleton().Parameters()["passes"].wasPassed = false;

  // Model with smaller number of passes.
  nodes = (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes();

  bindings::tests::CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  // Model with larger number of passes.
  SetInputParam("passes", 100);

  mlpackMain();

  // Check that model with larger number of passes has greater number of nodes.
  BOOST_REQUIRE_LT(nodes,
      (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes());
}

/**
 * Ensure that the root node has 2 children when splitting strategy is binary.
 */
BOOST_AUTO_TEST_CASE(HoeffdingBinarySplittingStrategyTest)
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

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("numeric_split_strategy", (string) "binary");
  SetInputParam("max_samples", 50);

  SetInputParam("confidence", 0.25);

  mlpackMain();

  // Check that number of children is 2.
  BOOST_REQUIRE_EQUAL(
      (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes()-1, 2);
}

/**
 * Ensure that the number of children varies with varying 'bins' in domingos.
 */
BOOST_AUTO_TEST_CASE(HoeffdingDomingosSplittingStrategyTest)
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("max_samples", 50);
  SetInputParam("bins", 20);

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;
  CLI::GetSingleton().Parameters()["max_samples"].wasPassed = false;
  CLI::GetSingleton().Parameters()["numeric_split_strategy"].wasPassed = false;
  CLI::GetSingleton().Parameters()["bins"].wasPassed = false;

  // Initial model.
  nodes = (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes();

  bindings::tests::CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("max_samples", 50);
  SetInputParam("bins", 10);

  mlpackMain();

  // Check that both models have different number of nodes.
  BOOST_CHECK_NE(
      (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes(), nodes);
}

/**
 * Ensure that the model doesn't split if observations before binning
 * is greater than total number of samples passed.
 */
BOOST_AUTO_TEST_CASE(HoeffdingBinningTest)
{
  arma::mat inputData;
  arma::mat modData;
  arma::Row<size_t> modLabels;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  modData = inputData.cols(0, 49);
  modLabels = labels.cols(0, 49);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, modData));
  SetInputParam("labels", std::move(modLabels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("min_samples", 10);

  // Set parameter to a value greater than number of samples.
  SetInputParam("observations_before_binning", 100);
  SetInputParam("confidence", 0.25);

  mlpackMain();

  // Check that no splitting has happened.
  BOOST_REQUIRE_EQUAL(
      (CLI::GetParam<HoeffdingTreeModel*>("output_model"))->NumNodes(), 1);
}

BOOST_AUTO_TEST_SUITE_END();
