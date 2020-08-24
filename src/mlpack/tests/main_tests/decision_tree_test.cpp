/**
 * @file tests/main_tests/decision_tree_test.cpp
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

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;
using namespace data;

struct DecisionTreeTestFixture
{
 public:
  DecisionTreeTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~DecisionTreeTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

void ResetDTSettings()
{
  IO::ClearSettings();
  IO::RestoreSettings(testName);
}

/**
 * Check that number of output points and
 * number of input points are equal.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionTreeOutputDimensionTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_rows == 3);
}

/**
 * Check that number of output points and number
 * of input points are equal for categorical dataset.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture,
                 "DecisionTreeCategoricalOutputDimensionTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_rows == 6);
}

/**
 * Make sure minimum leaf size is always a non-negative number.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionTreeMinimumLeafSizeTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("minimum_leaf_size", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure maximum depth is always a non-negative number.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture,
                 "DecisionTreeNonNegativeMaximumDepthTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("maximum_depth", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure minimum gain split is always a fraction in range [0,1].
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionMinimumGainSplitTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("minimum_gain_split", 1.5); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure minimum gain split produces regularised tree.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionRegularisationTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

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
  pred = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

  bindings::tests::CleanMemory();

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("minimum_gain_split", 0.5);

  // Input test data.
  SetInputParam("test", std::make_tuple(info, inputData));
  arma::Row<size_t> predRegularised;
  mlpackMain();
  predRegularised = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

  size_t count = 0;
  REQUIRE(pred.n_elem == predRegularised.n_elem);
  for (size_t i = 0; i < pred.n_elem; ++i)
  {
    if (pred[i] != predRegularised[i])
      count++;
  }

  REQUIRE(count > 0);
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionModelReuseTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

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
  predictions = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(IO::GetParam<arma::mat>("probabilities"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;
  IO::GetSingleton().Parameters()["weights"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model",
      std::move(IO::GetParam<DecisionTreeModel*>("output_model")));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_rows == 3);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(predictions, IO::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(probabilities, IO::GetParam<arma::mat>("probabilities"));
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionTreeTrainingVerTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  mlpackMain();

  DecisionTreeModel* model = IO::GetParam<DecisionTreeModel*>("output_model");
  IO::GetParam<DecisionTreeModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Input pre-trained model.
  SetInputParam("input_model", model);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that saved model trained on categorical dataset can be used again.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionModelCategoricalReuseTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    FAIL("Cannot load test dataset braziltourism_test.arff!");

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
  predictions = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(IO::GetParam<arma::mat>("probabilities"));

  DecisionTreeModel* model = IO::GetParam<DecisionTreeModel*>("output_model");
  IO::GetParam<DecisionTreeModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;
  IO::GetSingleton().Parameters()["weights"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model", model);

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("probabilities").n_rows == 6);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(predictions, IO::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(probabilities, IO::GetParam<arma::mat>("probabilities"));
}

/**
 * Check that different maximum depths give different results.
 */
TEST_CASE_METHOD(DecisionTreeTestFixture, "DecisionTreeMaximumDepthTest",
                 "[DecisionTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

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
  predictions = std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

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
                        IO::GetParam<arma::Row<size_t>>("predictions"));
}
