/**
 * @file tests/main_tests/dbscan_test.cpp
 * @author Nikhil Goel
 *
 * Test RUN_BINDING() of dbscan_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(DBSCANTestFixture);

/**
 * Check that number of output labels and number of input
 * points are equal.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANOutputDimensionTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  size_t inputSize = inputData.n_cols;

  SetInputParam("input", inputData);

  RUN_BINDING();

  // Check that number of predicted labels is equal to the input test points.
  REQUIRE(params.Get<arma::Row<size_t>>("assignments").n_cols == inputSize);
  REQUIRE(params.Get<arma::Row<size_t>>("assignments").n_rows == 1);
  REQUIRE(params.Get<arma::mat>("centroids").n_rows == 4);
  REQUIRE(params.Get<arma::mat>("centroids").n_cols >= 1);
}

/**
 * Check that radius of search(epsilon) is always non-negative.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANEpsilonTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) -0.5);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that minimum size of cluster is always non-negative.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANMinSizeTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("min_size", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that no point is labelled as noise point
 * when min_size is equal to 1.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANClusterNumberTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("min_size", (int) 1);
  SetInputParam("epsilon", (double) 0.1);

  RUN_BINDING();

  arma::Row<size_t> output;
  output = std::move(params.Get<arma::Row<size_t>>("assignments"));

  for (size_t i = 0; i < output.n_elem; ++i)
    REQUIRE(output[i] < inputData.n_cols);
}

/**
 * Check that the cluster assignment is different for different
 * values of epsilon.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANDiffEpsilonTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 1.0);

  RUN_BINDING();

  arma::Row<size_t> output1;
  output1 = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.5);

  RUN_BINDING();

  arma::Row<size_t> output2;
  output2 = std::move(params.Get<arma::Row<size_t>>("assignments"));

  REQUIRE(arma::accu(output1 != output2) > 1);
}

/**
 * Check that the cluster assignment is different for different
 * values of Min Size.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANDiffMinSizeTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.4);
  SetInputParam("min_size", (int) 5);

  RUN_BINDING();

  arma::Row<size_t> output1;
  output1 = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.5);
  SetInputParam("min_size", (int) 40);

  RUN_BINDING();

  arma::Row<size_t> output2;
  output2 = std::move(params.Get<arma::Row<size_t>>("assignments"));

  REQUIRE(arma::accu(output1 != output2) > 1);
}

/**
 * Check that the tree type should be from the given list of
 * tree types. ’kd’, ’r’, ’r-star’, ’x’, ’hilbert-r’, ’r-plus’,
 * ’r-plus-plus’, ’cover’, ’ball’.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANTreeTypeTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("tree_type", std::string("binary"));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that the assignment of cluster is same if
 * different tree type is used for search.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANDiffTreeTypeTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  // Tree type = kd tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("kd"));

  RUN_BINDING();

  arma::Row<size_t> kdOutput;
  kdOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree Type = r tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r"));

  RUN_BINDING();

  arma::Row<size_t> rOutput;
  rOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree type = r-star tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r-star"));

  RUN_BINDING();

  arma::Row<size_t> rStarOutput;
  rStarOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree Type = x tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("x"));

  RUN_BINDING();

  arma::Row<size_t> xOutput;
  xOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree Type = hilbert-r tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("hilbert-r"));

  RUN_BINDING();

  arma::Row<size_t> hilbertROutput;
  hilbertROutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree Type = r-plus tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r-plus"));

  RUN_BINDING();

  arma::Row<size_t> rPlusOutput;
  rPlusOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree Type = r-plus-plus tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("r-plus-plus"));

  RUN_BINDING();

  arma::Row<size_t> rPlusPlusOutput;
  rPlusPlusOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree Type = cover tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("cover"));

  RUN_BINDING();

  arma::Row<size_t> coverOutput;
  coverOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  // Tree Type = ball tree.

  SetInputParam("input", inputData);
  SetInputParam("tree_type", std::string("ball"));

  RUN_BINDING();

  arma::Row<size_t> ballOutput;
  ballOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

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
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANSingleTreeTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);

  RUN_BINDING();

  arma::Row<size_t> output;
  output = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  SetInputParam("input", inputData);
  SetInputParam("single_mode", true);

  RUN_BINDING();

  arma::Row<size_t> singleModeOutput;
  singleModeOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CheckMatrices(output, singleModeOutput);
}

/**
 * Check that the assignment of cluster is same if
 * single tree is used for search.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANNaiveSearchTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);

  RUN_BINDING();

  arma::Row<size_t> output;
  output = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  SetInputParam("input", inputData);
  SetInputParam("naive", true);

  RUN_BINDING();

  arma::Row<size_t> naiveOutput;
  naiveOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CheckMatrices(output, naiveOutput);
}

/**
 * Check that the assignment of cluster is different if
 * point selection policies are different.
 */
TEST_CASE_METHOD(DBSCANTestFixture, "DBSCANRandomSelectionFlagTest",
                 "[DBSCANMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.358);
  SetInputParam("min_size", 1);
  SetInputParam("selection_type", std::string("ordered"));

  RUN_BINDING();

  arma::Row<size_t> orderedOutput;
  orderedOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  CleanMemory();
  ResetSettings();

  SetInputParam("input", inputData);
  SetInputParam("epsilon", (double) 0.358);
  SetInputParam("min_size", 1);
  SetInputParam("selection_type", std::string("random"));

  RUN_BINDING();

  arma::Row<size_t> randomOutput;
  randomOutput = std::move(params.Get<arma::Row<size_t>>("assignments"));

  REQUIRE(arma::accu(orderedOutput != randomOutput) > 0);
}
