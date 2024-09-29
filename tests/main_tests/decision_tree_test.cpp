/**
 * @file tests/main_tests/decision_tree_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of decision_tree_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;
using namespace data;

BINDING_TEST_FIXTURE(DecisionTreeTestFixture);

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

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 3);
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

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 6);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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
  RUN_BINDING();
  pred = std::move(params.Get<arma::Row<size_t>>("predictions"));

  CleanMemory();

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  SetInputParam("minimum_gain_split", 0.5);

  // Input test data.
  SetInputParam("test", std::make_tuple(info, inputData));
  arma::Row<size_t> predRegularised;
  RUN_BINDING();
  predRegularised = std::move(params.Get<arma::Row<size_t>>("predictions"));

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

  RUN_BINDING();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));
  probabilities = std::move(params.Get<arma::mat>("probabilities"));
  DecisionTreeModel* m = params.Get<DecisionTreeModel*>("output_model");

  ResetSettings();

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model", m);

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 3);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(predictions, params.Get<arma::Row<size_t>>("predictions"));
  CheckMatrices(probabilities, params.Get<arma::mat>("probabilities"));
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

  RUN_BINDING();

  DecisionTreeModel* model = params.Get<DecisionTreeModel*>("output_model");
  params.Get<DecisionTreeModel*>("output_model") = NULL;

  CleanMemory();

  // Input pre-trained model.
  SetInputParam("input_model", model);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  RUN_BINDING();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));
  probabilities = std::move(params.Get<arma::mat>("probabilities"));

  DecisionTreeModel* model = params.Get<DecisionTreeModel*>("output_model");
  params.Get<DecisionTreeModel*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model", model);

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 6);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(predictions, params.Get<arma::Row<size_t>>("predictions"));
  CheckMatrices(probabilities, params.Get<arma::mat>("probabilities"));
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

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  arma::Row<size_t> predictions;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));

  CleanMemory();

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));
  SetInputParam("maximum_depth", (int) 2);

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  RUN_BINDING();

  CheckMatricesNotEqual(predictions,
                        params.Get<arma::Row<size_t>>("predictions"));
}
