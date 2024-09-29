/**
 * @file tests/main_tests/random_forest_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of random_forest_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(RandomForestTestFixture);

/**
 * Check that number of output points and number of input
 * points are equal and have appropriate number of classes.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestOutputDimensionTest",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(testData));

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
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestModelReuseTest",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", testData);

  RUN_BINDING();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));
  probabilities = std::move(params.Get<arma::mat>("probabilities"));

  // Reset passed parameters.
  RandomForestModel* m = params.Get<RandomForestModel*>("output_model");
  params.Get<RandomForestModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input trained model.
  SetInputParam("test", std::move(testData));
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
 * Make sure number of trees specified is always a positive number.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestNumOfTreesTest",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  SetInputParam("num_trees", (int) 0); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure minimum leaf size specified is always a positive number.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestMinimumLeafSizeTest",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  SetInputParam("minimum_leaf_size", (int) 0); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure maximum depth specified is always a positive number.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestMaximumDepthTest",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  SetInputParam("maximum_depth", (int) -1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure only one of training data or pre-trained model is passed, when
 * warm_start is not passed.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestTrainingVerTest",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  // Input pre-trained model.
  SetInputParam("input_model",
                params.Get<RandomForestModel*>("output_model"));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

template<typename TreeType>
inline bool CheckDifferentTrees(const TreeType& nodeA, const TreeType& nodeB)
{
  if (nodeA.SplitDimension() != nodeB.SplitDimension())
    return true;

  if (nodeA.NumChildren() != nodeB.NumChildren())
    return true;

  for (size_t i = 0; i < nodeA.NumChildren(); ++i)
    if (CheckDifferentTrees(nodeA.Child(i), nodeB.Child(i)))
      return true;

  return false;
}

/**
 * Ensure that the trees have different structure as the minimum leaf size is
 * changed.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestDiffMinLeafSizeTest",
                 "[RandomForestMainTest][BindingTests]")
{
  // Train for minimum leaf size 20.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("minimum_leaf_size", (int) 20);

  RUN_BINDING();

  // Calculate training accuracy.
  RandomForestModel* rf1 =
      std::move(params.Get<RandomForestModel*>("output_model"));
  params.Get<RandomForestModel*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  // Train for minimum leaf size 10.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("minimum_leaf_size", (int) 10);

  RUN_BINDING();

  RandomForestModel* rf2 =
      std::move(params.Get<RandomForestModel*>("output_model"));
  params.Get<RandomForestModel*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  // Train for minimum leaf size 1.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("minimum_leaf_size", (int) 1);

  RUN_BINDING();

  RandomForestModel* rf3 =
      std::move(params.Get<RandomForestModel*>("output_model"));
  params.Get<RandomForestModel*>("output_model") = NULL;

  // Check that each tree is different.
  for (size_t i = 0; i < rf1->rf.NumTrees(); ++i)
  {
    REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf2->rf.Tree(i)));
    REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf3->rf.Tree(i)));
  }

  delete rf1;
  delete rf2;
  delete rf3;
}

/**
 * Ensure that the number of trees are different when num_trees is specified
 * differently.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestDiffNumTreeTest",
                 "[RandomForestMainTest][BindingTests]")
{
  // Train for num_trees 1.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");

  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    FAIL("Cannot load labels for vc2__test_labels.txt");

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_trees", (int) 1);
  SetInputParam("minimum_leaf_size", (int) 1);

  RUN_BINDING();

  const size_t numTrees1 =
      params.Get<RandomForestModel*>("output_model")->rf.NumTrees();

  CleanMemory();
  ResetSettings();

  // Train for num_trees 5.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_trees", (int) 5);
  SetInputParam("minimum_leaf_size", (int) 1);

  RUN_BINDING();

  const size_t numTrees2 =
      params.Get<RandomForestModel*>("output_model")->rf.NumTrees();

  CleanMemory();
  ResetSettings();

  // Train for num_trees 10.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_trees", (int) 10);
  SetInputParam("minimum_leaf_size", (int) 1);

  RUN_BINDING();

  const size_t numTrees3 =
      params.Get<RandomForestModel*>("output_model")->rf.NumTrees();

  REQUIRE(numTrees1 != numTrees2);
  REQUIRE(numTrees2 != numTrees3);
}

/**
 * Ensure that the maximum_depth parameter makes a difference.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestDiffMaxDepthTest",
                 "[RandomForestMainTest][BindingTests]")
{
  // Train for minimum leaf size 20.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("maximum_depth", (int) 1);

  RUN_BINDING();

  // Calculate training accuracy.
  RandomForestModel* rf1 =
      std::move(params.Get<RandomForestModel*>("output_model"));
  params.Get<RandomForestModel*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("maximum_depth", (int) 2);

  RUN_BINDING();

  RandomForestModel* rf2 =
      std::move(params.Get<RandomForestModel*>("output_model"));
  params.Get<RandomForestModel*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  // Train for minimum leaf size 1.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("maximum_depth", (int) 3);

  RUN_BINDING();

  RandomForestModel* rf3 =
      std::move(params.Get<RandomForestModel*>("output_model"));
  params.Get<RandomForestModel*>("output_model") = NULL;

  // Check that each tree is different.
  for (size_t i = 0; i < rf1->rf.NumTrees(); ++i)
  {
    REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf2->rf.Tree(i)));
    REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf3->rf.Tree(i)));
  }

  delete rf1;
  delete rf2;
  delete rf3;
}

/**
 * Make sure that training and input_model are both passed when warm_start is
 * false.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestTrainingModelWarmStart",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  // Setting warm_start flag.
  SetInputParam("warm_start", false);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that model does gets trained on top of existing one when warm_start
 * and input_model are both passed.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestWarmStart",
                 "[RandomForestMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);

  RUN_BINDING();

  // Old number of trees in the model.
  size_t oldNumTrees =
      params.Get<RandomForestModel*>("output_model")->rf.NumTrees();

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("warm_start", true);

  // Input pre-trained model.
  SetInputParam("input_model",
                params.Get<RandomForestModel*>("output_model"));

  RUN_BINDING();

  size_t newNumTrees =
      params.Get<RandomForestModel*>("output_model")->rf.NumTrees();

  REQUIRE(oldNumTrees + 10 == newNumTrees);
}
