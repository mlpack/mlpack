/**
 * @file dbscan_test.cpp
 * @author Nikhil Goel
 *
 * Test mlpackMain() of dbscan_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "DBSCAN";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/dbscan/dbscan_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct DBSCANTestFixture
{
 public:
  DBSCANTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~DBSCANTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(DBSCANMainTest, DBSCANTestFixture);

/**
 * Check that number of output labels and number of input
 * points are equal.
 */
BOOST_AUTO_TEST_CASE(DBSCANOutputDimensionTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  size_t inputSize = inputData.n_cols;

  SetInputParam("input", inputData);

  mlpackMain();

  // Check that number of predicted labels is equal to the input test points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("assignments").n_cols,
                      inputSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("assignments").n_rows,
                      1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("centroids").n_rows, 4);
  BOOST_REQUIRE_GE(CLI::GetParam<arma::mat>("centroids").n_cols, 1);
}

/**
 * Check that radius of search(epsilon) is always non-negative.
 */
BOOST_AUTO_TEST_CASE(DBSCANEpsilonTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) -0.5);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that minimum size of cluster is always non-negative.
 */
BOOST_AUTO_TEST_CASE(DBSCANMinSizeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("min_size", (int) -1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that no point is labelled as noise point
 * when min_size is equal to 1.
 */
BOOST_AUTO_TEST_CASE(DBSCANClusterNumberTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("min_size", (int) 1);
  SetInputParam("epsilon", (double) 0.1);

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  for (size_t i = 0; i < output.n_elem; ++i)
    BOOST_REQUIRE_LT(output[i], inputData.n_cols);
}

/**
 * Check that the cluster assignment is different for different
 * values of epsilon.
 */
BOOST_AUTO_TEST_CASE(DBSCANDiffEpsilonTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 1.0);

  mlpackMain();

  arma::Row<size_t> output1;
  output1 = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["epsilon"].wasPassed = false;

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.5);

  mlpackMain();

  arma::Row<size_t> output2;
  output2 = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  BOOST_REQUIRE_GT(arma::accu(output1 != output2), 1);
}

/**
 * Check that the cluster assignment is different for different
 * values of Min Size.
 */
BOOST_AUTO_TEST_CASE(DBSCANDiffMinSizeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.4);
  SetInputParam("min_size", (int) 5);

  mlpackMain();

  arma::Row<size_t> output1;
  output1 = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["epsilon"].wasPassed = false;
  CLI::GetSingleton().Parameters()["min_size"].wasPassed = false;

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.5);
  SetInputParam("min_size", (int) 40);

  mlpackMain();

  arma::Row<size_t> output2;
  output2 = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  BOOST_REQUIRE_GT(arma::accu(output1 != output2), 1);
}

/**
 * Check that the tree type should be from the given list of
 * tree types. ’kd’, ’r’, ’r-star’, ’x’, ’hilbert-r’, ’r-plus’,
 * ’r-plus-plus’, ’cover’, ’ball’.
 */
BOOST_AUTO_TEST_CASE(DBSCANTreeTypeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("tree_type", std::string("binary"));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that the assignment of cluster is same if
 * different tree type is used for search.
 */
BOOST_AUTO_TEST_CASE(DBSCANDiffTreeTypeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  // Tree type = kd tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("kd"));

  mlpackMain();

  arma::Row<size_t> kdOutput;
  kdOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree Type = r tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r"));

  mlpackMain();

  arma::Row<size_t> rOutput;
  rOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree type = r-star tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r-star"));

  mlpackMain();

  arma::Row<size_t> rStarOutput;
  rStarOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree Type = x tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("x"));

  mlpackMain();

  arma::Row<size_t> xOutput;
  xOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree Type = hilbert-r tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("hilbert-r"));

  mlpackMain();

  arma::Row<size_t> hilbertROutput;
  hilbertROutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree Type = r-plus tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r-plus"));

  mlpackMain();

  arma::Row<size_t> rPlusOutput;
  rPlusOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree Type = r-plus-plus tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r-plus-plus"));

  mlpackMain();

  arma::Row<size_t> rPlusPlusOutput;
  rPlusPlusOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree Type = cover tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("cover"));

  mlpackMain();

  arma::Row<size_t> coverOutput;
  coverOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tree_type"].wasPassed = false;

  // Tree Type = ball tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("ball"));

  mlpackMain();

  arma::Row<size_t> ballOutput;
  ballOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  CheckMatrices(kdOutput, rOutput);
  CheckMatrices(kdOutput, rStarOutput);
  CheckMatrices(kdOutput, xOutput);
  CheckMatrices(kdOutput, hilbertROutput);
  CheckMatrices(kdOutput, rPlusOutput);
  CheckMatrices(kdOutput, rPlusPlusOutput);
  CheckMatrices(kdOutput, coverOutput);
  CheckMatrices(kdOutput, ballOutput);
}

/**
 * Check that the assignment of cluster is same if
 * single tree is used for search.
 */
BOOST_AUTO_TEST_CASE(DBSCANSingleTreeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;

  SetInputParam("input", inputData);
  SetInputParam("single_mode", true);

  mlpackMain();

  arma::Row<size_t> singleModeOutput;
  singleModeOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  CheckMatrices(output, singleModeOutput);
}

/**
 * Check that the assignment of cluster is same if
 * single tree is used for search.
 */
BOOST_AUTO_TEST_CASE(DBSCANNaiveSearchTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);

  mlpackMain();

  arma::Row<size_t> output;
  output = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;

  SetInputParam("input", inputData);
  SetInputParam("naive", true);

  mlpackMain();

  arma::Row<size_t> naiveOutput;
  naiveOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  CheckMatrices(output, naiveOutput);
}

/**
 * Check that the assignment of cluster is different if
 * point selection policies are different.
 */
BOOST_AUTO_TEST_CASE(DBSCANRandomSelectionFlagTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.358);
  SetInputParam("min_size", 1);
  SetInputParam("selection_type", std::string("ordered"));

  mlpackMain();

  arma::Row<size_t> orderedOutput;
  orderedOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["epsilon"].wasPassed = false;
  CLI::GetSingleton().Parameters()["min_size"].wasPassed = false;
  CLI::GetSingleton().Parameters()["selection_type"].wasPassed = false;

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.358);
  SetInputParam("min_size", 1);
  SetInputParam("selection_type", std::string("random"));

  mlpackMain();

  arma::Row<size_t> randomOutput;
  randomOutput = std::move(CLI::GetParam<arma::Row<size_t>>("assignments"));

  BOOST_REQUIRE_GT(arma::accu(orderedOutput != randomOutput), 0);
}

BOOST_AUTO_TEST_SUITE_END();
