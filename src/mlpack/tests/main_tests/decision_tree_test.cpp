/**
 * @file decision_tree_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of decision_tree_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "DecisionTree";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/decision_tree/decision_tree_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;
using namespace data;

struct DecisionTreeTestFixture
{
 public:
  DecisionTreeTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~DecisionTreeTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

void ResetDTSettings()
{
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(DecisionTreeMainTest,
                         DecisionTreeTestFixture);

/**
 * Check that number of output points and
 * number of input points are equal.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeOutputDimensionTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 3);
}

/**
 * Check that number of output points and number
 * of input points are equal for categorical dataset.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeCategoricalOutputDimensionTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    BOOST_FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 6);
}

/**
 * Make sure minimum leaf size is always a non-negative number.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeMinimumLeafSizeTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("minimum_leaf_size", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure maximum depth is always a non-negative number.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeNonNegativeMaximumDepthTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("maximum_depth", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure minimum gain split is always a fraction in range [0,1].
 */
BOOST_AUTO_TEST_CASE(DecisionMinimumGainSplitTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::move(std::make_tuple(info, inputData)));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("minimum_gain_split", 1.5); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure minimum gain split produces regularised tree.
 */
BOOST_AUTO_TEST_CASE(DecisionRegularisationTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", labels);
  SetInputParam("weights", weights);

  SetInputParam("minimum_gain_split", 1e-7);

  // Input test data.
  SetInputParam("test", std::make_tuple(info, inputData));
  arma::Row<size_t> pred;
  mlpackMain();
  pred = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  bindings::tests::CleanMemory();

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("minimum_gain_split", 0.01);

  // Input test data.
  SetInputParam("test", std::move(std::make_tuple(info, inputData)));
  arma::Row<size_t> predRegularised;
  mlpackMain();
  predRegularised = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  size_t count = 0;
  BOOST_REQUIRE_EQUAL(pred.n_elem, predRegularised.n_elem);
  for (size_t i = 0; i < pred.n_elem; ++i)
  {
    if (pred[i] != predRegularised[i])
      count++;
  }

  BOOST_REQUIRE_GT(count, 0);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(DecisionModelReuseTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["weights"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model",
      std::move(CLI::GetParam<DecisionTreeModel*>("output_model")));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 3);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(probabilities, CLI::GetParam<arma::mat>("probabilities"));
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeTrainingVerTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  mlpackMain();

  DecisionTreeModel* model = CLI::GetParam<DecisionTreeModel*>("output_model");
  CLI::GetParam<DecisionTreeModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Input pre-trained model.
  SetInputParam("input_model", model);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that saved model trained on categorical dataset can be used again.
 */
BOOST_AUTO_TEST_CASE(DecisionModelCategoricalReuseTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    BOOST_FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    BOOST_FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  DecisionTreeModel* model = CLI::GetParam<DecisionTreeModel*>("output_model");
  CLI::GetParam<DecisionTreeModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["weights"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model", model);

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  BOOST_REQUIRE_EQUAL(
      CLI::GetParam<arma::Row<size_t>>("predictions").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 6);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(probabilities, CLI::GetParam<arma::mat>("probabilities"));
}

/**
 * Check that different maximum depths give different results.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeMaximumDepthTest)
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", labels);
  SetInputParam("weights", weights);
  SetInputParam("maximum_depth", (int) 0);

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  arma::Row<size_t> predictions;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  bindings::tests::CleanMemory();

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));
  SetInputParam("maximum_depth", (int) 2);

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  CheckMatricesNotEqual(predictions,
                        CLI::GetParam<arma::Row<size_t>>("predictions"));
}

BOOST_AUTO_TEST_SUITE_END();
