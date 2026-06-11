/**
 * @file tests/main_tests/hoeffding_tree_train_test.cpp
 * @author Haritha Nair
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of hoeffding_tree_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(HoeffdingTreeTrainTestFixture);

/**
 * Check that number of output points and
 * number of input points are equal.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTreeTrainOutputDimensionTest",
                 "[HoeffdingTreeTrainMainTest][BindingsTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  DatasetInfo info(D);
  SetInputParam("training", std::make_tuple(info, std::move(trainX)));
  SetInputParam("labels", std::move(trainY));
  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  arma::Row<size_t> preds;
  arma::mat::fixed<2, 1> testX = { 0.123, 0.456 };
  auto model = params.Get<HoeffdingTreeModel*>("output_model");
  model->Classify(testX, preds);
  REQUIRE(preds.n_elem == 1);

  arma::rowvec probs;
  model->Classify(testX, preds, probs);
  REQUIRE(preds.n_elem == 1);
  REQUIRE(probs.n_elem == 1);
}

/**
 * Check that number of output points and number
 * of input points are equal for categorical dataset.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTreeTrainCategoricalOutputDimensionTest",
                 "[HoeffdingTreeTrainMainTest][BindingsTests]")
{
  arma::mat inputData;
  TextOptions opts;
  opts.Categorical() = true;

  if (!Load("braziltourism.arff", inputData, opts))
    FAIL("Cannot load train dataset braziltourism.arff!");

  arma::Row<size_t> labels;
  if (!Load("braziltourism_labels.txt", labels))
    FAIL("Cannot load labels for braziltourism_labels.txt");

  arma::mat testData;
  if (!Load("braziltourism_test.arff", testData, opts))
    FAIL("Cannot load test dataset braziltourism_test.arff!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::make_tuple(opts.DatasetInfo(), inputData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  // Check that number of output points are equal to number of input points.
  arma::Row<size_t> preds;
  auto model = params.Get<HoeffdingTreeModel*>("output_model");
  model->Classify(testData, preds);
  REQUIRE(preds.n_elem == testSize);

  arma::rowvec probs;
  model->Classify(testData, preds, probs);
  REQUIRE(preds.n_elem == testSize);
  REQUIRE(probs.n_elem == testSize);
}

/**
 * Ensure that small min_samples creates larger model.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTrainMinSamplesTest",
                 "[HoeffdingTreeTrainMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::Row<size_t> labels;
  TextOptions opts;
  opts.Categorical() = true;

  if (!Load("vc2.csv", inputData, opts)) FAIL("Cannot load vc2.csv!");
  if (!Load("vc2_labels.txt", labels)) FAIL("Cannot load vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::make_tuple(opts.DatasetInfo(), inputData));
  SetInputParam("labels", labels);
  SetInputParam("min_samples", 10);
  SetInputParam("confidence", 0.25);
  RUN_BINDING();
  auto nodes = params.Get<HoeffdingTreeModel*>("output_model")->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  // Input training data.
  SetInputParam("training",
      std::make_tuple(opts.DatasetInfo(), std::move(inputData)));
  SetInputParam("labels", std::move(labels));
  SetInputParam("min_samples", 2000);
  SetInputParam("confidence", 0.25);
  RUN_BINDING();

  // Check that small min_samples creates larger model.
  REQUIRE(params.Get<HoeffdingTreeModel*>("output_model")->NumNodes() < nodes);
}

/**
 * Ensure that large max_samples creates smaller model.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTrainMaxSamplesTest",
                 "[HoeffdingTreeTrainMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::Row<size_t> labels;
  TextOptions opts;
  opts.Categorical() = true;

  if (!Load("vc2.csv", inputData, opts)) FAIL("Cannot load vc2.csv!");
  if (!Load("vc2_labels.txt", labels)) FAIL("Cannot load vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::make_tuple(opts.DatasetInfo(), inputData));
  SetInputParam("labels", labels);
  SetInputParam("max_samples", 50000);
  SetInputParam("confidence", 0.95);
  RUN_BINDING();
  auto nodes = params.Get<HoeffdingTreeModel*>("output_model")->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  // Input training data.
  SetInputParam("training",
      std::make_tuple(opts.DatasetInfo(), std::move(inputData)));
  SetInputParam("labels", std::move(labels));
  SetInputParam("max_samples", 5);
  SetInputParam("confidence", 0.95);
  RUN_BINDING();

  // Check that large max_samples creates smaller model.
  REQUIRE(nodes < params.Get<HoeffdingTreeModel*>("output_model")->NumNodes());
}

/**
 * Ensure that small confidence value creates larger model.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTrainConfidenceTest",
                 "[HoeffdingTreeTrainMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::Row<size_t> labels;
  TextOptions opts;
  opts.Categorical() = true;

  if (!Load("vc2.csv", inputData, opts)) FAIL("Cannot load vc2.csv!");
  if (!Load("vc2_labels.txt", labels)) FAIL("Cannot load vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::make_tuple(opts.DatasetInfo(), inputData));
  SetInputParam("labels", labels);
  SetInputParam("confidence", 0.95); // high confidence
  RUN_BINDING();

  // Model with high confidence.
  auto nodes = params.Get<HoeffdingTreeModel*>("output_model")->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  // Input training data.
  SetInputParam("training",
      std::make_tuple(opts.DatasetInfo(), std::move(inputData)));
  SetInputParam("labels", std::move(labels));
  SetInputParam("confidence", 0.25); // low confidence
  RUN_BINDING();

  // Check that higher confidence creates smaller tree.
  REQUIRE(nodes < params.Get<HoeffdingTreeModel*>("output_model")->NumNodes());
}

/**
 * Ensure that the root node has 2 children when splitting strategy is binary.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTrainBinarySplittingStrategyTest",
                 "[HoeffdingTreeTrainMainTest][BindingsTests]")
{
  arma::mat inputData;
  arma::Row<size_t> labels;
  TextOptions opts;
  opts.Categorical() = true;

  if (!Load("vc2.csv", inputData, opts)) FAIL("Cannot load vc2.csv!");
  if (!Load("vc2_labels.txt", labels)) FAIL("Cannot load vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::make_tuple(opts.DatasetInfo(), inputData));
  SetInputParam("labels", labels);
  SetInputParam("numeric_split_strategy", (string) "binary");
  SetInputParam("max_samples", 50);
  SetInputParam("confidence", 0.25);
  RUN_BINDING();

  // Check that number of children is 2.
  REQUIRE(params.Get<HoeffdingTreeModel*>("output_model")->NumNodes() - 1 == 2);
}

/**
 * Ensure that the number of children varies with varying 'bins' in domingos.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTrainDomingosSplittingStrategyTest",
                 "[HoeffdingTreeTrainMainTest][BindingsTests]")
{
  arma::mat inputData;
  arma::Row<size_t> labels;
  TextOptions opts;
  opts.Categorical() = true;

  if (!Load("vc2.csv", inputData, opts)) FAIL("Cannot load vc2.csv!");
  if (!Load("vc2_labels.txt", labels)) FAIL("Cannot load vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::make_tuple(opts.DatasetInfo(), inputData));
  SetInputParam("labels", labels);
  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("max_samples", 50);
  SetInputParam("bins", 20);
  RUN_BINDING();

  // Initial model.
  auto nodes = params.Get<HoeffdingTreeModel*>("output_model")->NumNodes();

  // Reset passed parameters.
  ResetSettings();
  CleanMemory();

  // Input training data.
  SetInputParam("training",
      std::make_tuple(opts.DatasetInfo(), std::move(inputData)));
  SetInputParam("labels", std::move(labels));
  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("max_samples", 50);
  SetInputParam("bins", 10);
  RUN_BINDING();

  // Check that both models have different number of nodes.
  REQUIRE(params.Get<HoeffdingTreeModel*>("output_model")->NumNodes() != nodes);
}


/**
 * Ensure that the model doesn't split if observations before binning
 * is greater than total number of samples passed.
 */
TEST_CASE_METHOD(HoeffdingTreeTrainTestFixture,
                 "HoeffdingTrainBinningTest",
                 "[HoeffdingTreeTrainMainTest][BindingsTests]")
{
  arma::mat inputData;
  arma::mat modData;
  arma::Row<size_t> labels;
  arma::Row<size_t> modLabels;
  TextOptions opts;
  opts.Categorical() = true;

  if (!Load("vc2.csv", inputData, opts)) FAIL("Cannot load vc2.csv!");
  if (!Load("vc2_labels.txt", labels)) FAIL("Cannot load vc2_labels.txt");

  modData = inputData.cols(0, 49);
  modLabels = labels.cols(0, 49);

  // Input training data.
  SetInputParam("training",
      std::make_tuple(opts.DatasetInfo(), std::move(modData)));
  SetInputParam("labels", std::move(modLabels));

  SetInputParam("numeric_split_strategy", (string) "domingos");
  SetInputParam("min_samples", 10);

  // Set parameter to a value greater than number of samples.
  SetInputParam("observations_before_binning", 100);
  SetInputParam("confidence", 0.25);

  RUN_BINDING();

  // Check that no splitting has happened.
  REQUIRE(params.Get<HoeffdingTreeModel*>("output_model")->NumNodes() == 1);
}
