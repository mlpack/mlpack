/**
 * @file tests/main_tests/hoeffding_tree_test.cpp
 * @author Haritha Nair
 *
 * Test RUN_BINDING() of hoeffding_tree_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;
using namespace data;

BINDING_TEST_FIXTURE(HoeffdingTreeTestFixture);

/**
 * Check that number of output points and
 * number of input points are equal.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingTreeOutputDimensionTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 1);
}

/**
 * Check that number of output points and number
 * of input points are equal for categorical dataset.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture,
                 "HoeffdingTreeCategoricalOutputDimensionTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 1);
}

/**
 * Check whether providing labels explicitly and extracting from last
 * dimension give the same output.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingTreeLabelLessTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Append labels to the training set.
  inputData.resize(inputData.n_rows+1, inputData.n_cols);
  for (size_t i = 0; i < inputData.n_cols; ++i)
    inputData(inputData.n_rows-1, i) = labels[i];

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 1);

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));
  probabilities = std::move(params.Get<arma::mat>("probabilities"));

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  inputData.shed_row(inputData.n_rows - 1);

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("test", std::make_tuple(info, testData));
  // Pass Labels.
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 1);

  // Check that initial and current predictions are same.
  CheckMatrices(
      predictions, params.Get<arma::Row<size_t>>("predictions"));
  CheckMatrices(
      probabilities, params.Get<arma::mat>("probabilities"));
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingModelReuseTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  RUN_BINDING();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));
  probabilities = std::move(params.Get<arma::mat>("probabilities"));

  // Reset passed parameters.
  HoeffdingTreeModel* m = params.Get<HoeffdingTreeModel*>("output_model");
  ResetSettings();

  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model", m);

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 1);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(
      predictions, params.Get<arma::Row<size_t>>("predictions"));
  CheckMatrices(
      probabilities, params.Get<arma::mat>("probabilities"));

  ResetSettings();
  delete m;
}

/**
 * Ensure that saved model trained on categorical dataset can be used again.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingModelCategoricalReuseTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("braziltourism.arff", inputData, info))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!data::Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  arma::mat testData;
  if (!data::Load("braziltourism_test.arff", testData, info))
    FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  RUN_BINDING();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));
  probabilities = std::move(params.Get<arma::mat>("probabilities"));

  // Reset passed parameters.
  HoeffdingTreeModel* m = params.Get<HoeffdingTreeModel*>("output_model");
  ResetSettings();

  if (!data::Load("braziltourism_test.arff", testData, info))
    FAIL("Cannot load test dataset braziltourism_test.arff!");

  // Input trained model.
  SetInputParam("test", std::make_tuple(info, testData));
  SetInputParam("input_model", m);

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_cols == testSize);
  REQUIRE(params.Get<arma::mat>("probabilities").n_cols == testSize);

  // Check number of output rows equals 1 for probabilities and predictions.
  REQUIRE(params.Get<arma::Row<size_t>>("predictions").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("probabilities").n_rows == 1);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(
      predictions, params.Get<arma::Row<size_t>>("predictions"));
  CheckMatrices(
      probabilities, params.Get<arma::mat>("probabilities"));

  ResetSettings();
  delete m;
}

/**
 * Ensure that small min_samples creates larger model.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingMinSamplesTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("min_samples", 10);
  SetInputParam("confidence", 0.25);

  RUN_BINDING();

  nodes = (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("min_samples", 2000);
  SetInputParam("confidence", 0.25);

  RUN_BINDING();

  // Check that small min_samples creates larger model.
  REQUIRE((params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes() <
      (size_t) nodes);
}

/**
 * Ensure that large max_samples creates smaller model.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingMaxSamplesTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("max_samples", 50000);
  SetInputParam("confidence", 0.95);

  RUN_BINDING();

  nodes = (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("max_samples", 5);
  SetInputParam("confidence", 0.95);

  RUN_BINDING();

  // Check that large max_samples creates smaller model.
  REQUIRE((size_t) nodes <
      (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes());
}

/**
 * Ensure that small confidence value creates larger model.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingConfidenceTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("confidence", 0.95);

  RUN_BINDING();

  // Model with high confidence.
  nodes = (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  // Model with low confidence.
  SetInputParam("confidence", 0.25);

  RUN_BINDING();
  // Check that higher confidence creates smaller tree.
  REQUIRE((size_t) nodes <
      (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes());
}

/**
 * Ensure that large number of passes creates larger model.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingPassesTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("passes", 1);

  RUN_BINDING();

  // Model with smaller number of passes.
  nodes = (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  // Model with larger number of passes.
  SetInputParam("passes", 100);

  RUN_BINDING();

  // Check that model with larger number of passes has greater number of nodes.
  REQUIRE((size_t) nodes <
      (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes());
}

/**
 * Ensure that the root node has 2 children when splitting strategy is binary.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture,
                 "HoeffdingBinarySplittingStrategyTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("numeric_split_strategy", (string) "binary");
  SetInputParam("max_samples", 50);

  SetInputParam("confidence", 0.25);

  RUN_BINDING();

  // Check that number of children is 2.
  REQUIRE(
      (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes() - 1 == 2);
}

/**
 * Ensure that the number of children varies with varying 'bins' in domingos.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture,
                 "HoeffdingDomingosSplittingStrategyTest",
                 "[HoeffdingTreeMainTest][BindingTest]")
{
  arma::mat inputData;
  DatasetInfo info;
  int nodes;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("max_samples", 50);
  SetInputParam("bins", 20);

  RUN_BINDING();

  // Initial model.
  nodes = (params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

  // Input training data.
  SetInputParam("training", std::make_tuple(info, inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::make_tuple(info, testData));

  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("max_samples", 50);
  SetInputParam("bins", 10);

  RUN_BINDING();

  // Check that both models have different number of nodes.
  CHECK((params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes() !=
      (size_t) nodes);
}

/**
 * Ensure that the model doesn't split if observations before binning
 * is greater than total number of samples passed.
 */
TEST_CASE_METHOD(HoeffdingTreeTestFixture, "HoeffdingBinningTest",
                 "[HoeffdingTreeMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::mat modData;
  arma::Row<size_t> modLabels;
  DatasetInfo info;
  if (!data::Load("vc2.csv", inputData, info))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData, info))
    FAIL("Cannot load test dataset vc2.csv!");

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

  RUN_BINDING();

  // Check that no splitting has happened.
  REQUIRE((params.Get<HoeffdingTreeModel*>("output_model"))->NumNodes()
      == 1);
}
