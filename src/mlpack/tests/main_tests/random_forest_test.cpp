/**
 * @file random_forest_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of random_forest_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "RandomForest";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/random_forest/random_forest_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct RandomForestTestFixture
{
 public:
  RandomForestTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~RandomForestTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(RandomForestMainTest, RandomForestTestFixture);

/**
 * Check that number of output points and number of input
 * points are equal and have appropriate number of classes.
 */
BOOST_AUTO_TEST_CASE(RandomForestOutputDimensionTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
                      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predictions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
                      1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 3);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(RandomForestModelReuseTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Input test data.
  SetInputParam("test", testData);

  mlpackMain();

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  predictions = std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));
  probabilities = std::move(CLI::GetParam<arma::mat>("probabilities"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  // Input trained model.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                CLI::GetParam<RandomForestModel*>("output_model"));

  mlpackMain();

  // Check that number of output points are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
                      testSize);

  // Check number of output rows equals number of classes in case of
  // probabilities and 1 for predicitions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_rows,
                      1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_rows, 3);

  // Check that initial predictions and predictions using saved model are same.
  CheckMatrices(predictions, CLI::GetParam<arma::Row<size_t>>("predictions"));
  CheckMatrices(probabilities, CLI::GetParam<arma::mat>("probabilities"));
}

/**
 * Make sure number of trees specified is always a positive number.
 */
BOOST_AUTO_TEST_CASE(RandomForestNumOfTreesTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  SetInputParam("num_trees", (int) 0); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure minimum leaf size specified is always a positive number.
 */
BOOST_AUTO_TEST_CASE(RandomForestMinimumLeafSizeTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  SetInputParam("minimum_leaf_size", (int) 0); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure maximum depth specified is always a positive number.
 */
BOOST_AUTO_TEST_CASE(RandomForestMaximumDepthTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  SetInputParam("maximum_depth", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure only one of training data or pre-trained model is passed.
 */
BOOST_AUTO_TEST_CASE(RandomForestTrainingVerTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
                CLI::GetParam<RandomForestModel*>("output_model"));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
BOOST_AUTO_TEST_CASE(RandomForestDiffMinLeafSizeTest)
{
  // Train for minimum leaf size 20.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("minimum_leaf_size", (int) 20);

  mlpackMain();

  // Calculate training accuracy.
  RandomForestModel* rf1 =
      std::move(CLI::GetParam<RandomForestModel*>("output_model"));
  CLI::GetParam<RandomForestModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Train for minimum leaf size 10.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("minimum_leaf_size", (int) 10);

  mlpackMain();

  RandomForestModel* rf2 =
      std::move(CLI::GetParam<RandomForestModel*>("output_model"));
  CLI::GetParam<RandomForestModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Train for minimum leaf size 1.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("minimum_leaf_size", (int) 1);

  mlpackMain();

  RandomForestModel* rf3 =
      std::move(CLI::GetParam<RandomForestModel*>("output_model"));
  CLI::GetParam<RandomForestModel*>("output_model") = NULL;

  // Check that each tree is different.
  for (size_t i = 0; i < rf1->rf.NumTrees(); ++i)
  {
    BOOST_REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf2->rf.Tree(i)));
    BOOST_REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf3->rf.Tree(i)));
  }

  delete rf1;
  delete rf2;
  delete rf3;
}

/**
 * Ensure that the number of trees are different when num_trees is specified
 * differently.
 */
BOOST_AUTO_TEST_CASE(RandomForestDiffNumTreeTest)
{
  // Train for num_trees 1.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset vc2_test.csv!");

  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    BOOST_FAIL("Cannot load labels for vc2__test_labels.txt");

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_trees", (int) 1);
  SetInputParam("minimum_leaf_size", (int) 1);

  mlpackMain();

  const size_t numTrees1 =
      CLI::GetParam<RandomForestModel*>("output_model")->rf.NumTrees();

  bindings::tests::CleanMemory();

  // Train for num_trees 5.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_trees", (int) 5);
  SetInputParam("minimum_leaf_size", (int) 1);

  mlpackMain();

  const size_t numTrees2 =
      CLI::GetParam<RandomForestModel*>("output_model")->rf.NumTrees();

  bindings::tests::CleanMemory();

  // Train for num_trees 10.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_trees", (int) 10);
  SetInputParam("minimum_leaf_size", (int) 1);

  mlpackMain();

  const size_t numTrees3 =
      CLI::GetParam<RandomForestModel*>("output_model")->rf.NumTrees();

  BOOST_REQUIRE_NE(numTrees1, numTrees2);
  BOOST_REQUIRE_NE(numTrees2, numTrees3);
}

/**
 * Ensure that the maximum_depth parameter makes a difference.
 */
BOOST_AUTO_TEST_CASE(RandomForestDiffMaxDepthTest)
{
  // Train for minimum leaf size 20.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("maximum_depth", (int) 1);

  mlpackMain();

  // Calculate training accuracy.
  RandomForestModel* rf1 =
      std::move(CLI::GetParam<RandomForestModel*>("output_model"));
  CLI::GetParam<RandomForestModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("maximum_depth", (int) 2);

  mlpackMain();

  RandomForestModel* rf2 =
      std::move(CLI::GetParam<RandomForestModel*>("output_model"));
  CLI::GetParam<RandomForestModel*>("output_model") = NULL;

  bindings::tests::CleanMemory();

  // Train for minimum leaf size 1.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("labels", labels);
  SetInputParam("maximum_depth", (int) 3);

  mlpackMain();

  RandomForestModel* rf3 =
      std::move(CLI::GetParam<RandomForestModel*>("output_model"));
  CLI::GetParam<RandomForestModel*>("output_model") = NULL;

  // Check that each tree is different.
  for (size_t i = 0; i < rf1->rf.NumTrees(); ++i)
  {
    BOOST_REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf2->rf.Tree(i)));
    BOOST_REQUIRE(CheckDifferentTrees(rf1->rf.Tree(i), rf3->rf.Tree(i)));
  }

  delete rf1;
  delete rf2;
  delete rf3;
}

BOOST_AUTO_TEST_SUITE_END();
