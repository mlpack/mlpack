/**
 * @file tests/main_tests/preprocess_split_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of preprocess_split_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/preprocess/preprocess_split_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(PreprocessSplitTestFixture);

/**
 * Check that desired output dimensions are received for both input data and
 * labels.
 */
TEST_CASE_METHOD(PreprocessSplitTestFixture, "PreprocessSplitDimensionTest",
                 "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  arma::Mat<size_t> labels;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  // Store size of input dataset.
  int inputSize  = inputData.n_cols;
  int labelSize  = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  // Input test_ratio.
  SetInputParam("test_ratio", (double) 0.1);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols ==
      std::ceil(0.9 * inputSize));
  REQUIRE(params.Get<arma::mat>("test").n_cols ==
      std::floor(0.1 * inputSize));

  REQUIRE(
      params.Get<arma::Mat<size_t>>("training_labels").n_cols ==
      std::ceil(0.9 * labelSize));
  REQUIRE(params.Get<arma::Mat<size_t>>("test_labels").n_cols ==
      std::floor(0.1 * labelSize));
}

/**
 * Check that desired output dimensions are received for the input data when
 * labels are not provided.
 */
TEST_CASE_METHOD(
    PreprocessSplitTestFixture,
    "PreprocessSplitLabelLessDimensionTest",
    "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  // Store size of input dataset.
  int inputSize  = inputData.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));

  // Input test_ratio.
  SetInputParam("test_ratio", (double) 0.1);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols ==
      std::ceil(0.9 * inputSize));
  REQUIRE(params.Get<arma::mat>("test").n_cols ==
      std::floor(0.1 * inputSize));
}

/**
 * Ensure that test ratio is always a non-negative number.
 */
TEST_CASE_METHOD(PreprocessSplitTestFixture, "PreprocessSplitTestRatioTest",
                 "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  arma::Mat<size_t> labels;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) -0.2);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that if test size is 0 then train consist of whole input data.
 */
TEST_CASE_METHOD(
    PreprocessSplitTestFixture, "PreprocessSplitZeroTestRatioTest",
    "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  arma::Mat<size_t> labels;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  // Store size of input dataset.
  int inputSize = inputData.n_cols;
  int labelSize = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) 0.0);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols ==
      (arma::uword) inputSize);
  REQUIRE(params.Get<arma::mat>("test").n_cols == 0);

  REQUIRE(params.Get<arma::Mat<size_t>>("training_labels").n_cols ==
      (arma::uword) labelSize);
  REQUIRE(params.Get<arma::Mat<size_t>>("test_labels").n_cols == 0);
}

/**
 * Check that if test size is 1 then test consist of whole input data.
 */
TEST_CASE_METHOD(
    PreprocessSplitTestFixture, "PreprocessSplitUnityTestRatioTest",
    "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  arma::Mat<size_t> labels;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  // Store size of input dataset.
  int inputSize = inputData.n_cols;
  int labelSize = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) 1.0);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols == 0);
  REQUIRE(params.Get<arma::mat>("test").n_cols == (arma::uword) inputSize);

  REQUIRE(params.Get<arma::Mat<size_t>>("training_labels").n_cols == 0);
  REQUIRE(params.Get<arma::Mat<size_t>>("test_labels").n_cols ==
      (arma::uword) labelSize);
}

/**
 * Check shuffle_data flag is working as expected.
 */
TEST_CASE_METHOD(
    PreprocessSplitTestFixture, "PreprocessSplitLabelShuffleDataTest",
    "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");

  // Store size of input dataset.
  int inputSize  = inputData.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", inputData);

  // Input test_ratio.
  SetInputParam("test_ratio", (double) 0.1);
  SetInputParam("no_shuffle", true);
  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols ==
      std::ceil(0.9 * inputSize));
  REQUIRE(params.Get<arma::mat>("test").n_cols ==
      std::floor(0.1 * inputSize));

  arma::mat concat = arma::join_rows(params.Get<arma::mat>("training"),
      params.Get<arma::mat>("test"));
  CheckMatrices(inputData, concat);
}

/**
 * Check that if test size is 0 then train consist of whole input data when
 * stratifying.
 */
TEST_CASE_METHOD(
    PreprocessSplitTestFixture, "PreprocessStratifiedSplitZeroTestRatioTest",
    "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  arma::Mat<size_t> labels;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  // Store size of input dataset.
  int inputSize = inputData.n_cols;
  int labelSize = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) 0.0);
  SetInputParam("stratify_data", true);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols ==
      (arma::uword) inputSize);
  REQUIRE(params.Get<arma::mat>("test").n_cols == 0);

  REQUIRE(params.Get<arma::Mat<size_t>>("training_labels").n_cols ==
      (arma::uword) labelSize);
  REQUIRE(params.Get<arma::Mat<size_t>>("test_labels").n_cols == 0);
}

/**
 * Check that if test size is 1 then test consist of whole input data when
 * stratifying.
 */
TEST_CASE_METHOD(
    PreprocessSplitTestFixture, "PreprocessStratifiedSplitUnityTestRatioTest",
    "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  arma::Mat<size_t> labels;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  // Store size of input dataset.
  int inputSize = inputData.n_cols;
  int labelSize = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) 1.0);
  SetInputParam("stratify_data", true);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols == 0);
  REQUIRE(params.Get<arma::mat>("test").n_cols == (arma::uword) inputSize);

  REQUIRE(params.Get<arma::Mat<size_t>>("training_labels").n_cols == 0);
  REQUIRE(params.Get<arma::Mat<size_t>>("test_labels").n_cols ==
      (arma::uword) labelSize);
}

/**
 * Checking label wise counts to ensure data is stratified when passing the
 * stratify_data param.
 *
 * The vc2 dataset labels file contains 40 0s, 100 1s, and 67 2s.
 * Considering a test ratio of 0.3,
 * Number of 0s in the test set lables =  12 ( floor(40 * 0.3) = floor(12) ).
 * Number of 1s in the test set labels =  30 ( floor(100 * 0.3) = floor(30) ).
 * Number of 2s in the test set labels =  20 ( floor(67 * 0.3) = floor(20.1) ).
 * Total points in the test set = 62 ( 12 + 30 + 20 ).
 */
TEST_CASE_METHOD(
    PreprocessSplitTestFixture, "PreprocessStratifiedSplitTest",
    "[PreprocessSplitMainTest][BindingTests]")
{
  // Load custom dataset.
  arma::mat inputData;
  arma::Mat<size_t> labels;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load train dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) 0.3);
  SetInputParam("stratify_data", true);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("training").n_cols == 145);
  REQUIRE(params.Get<arma::mat>("test").n_cols == 62);

  // Checking for specific label counts in the output.
  REQUIRE(static_cast<uvec>(find(
      params.Get<arma::Mat<size_t>>("training_labels") == 0)).n_rows == 28);
  REQUIRE(static_cast<uvec>(find(
      params.Get<arma::Mat<size_t>>("training_labels") == 1)).n_rows == 70);
  REQUIRE(static_cast<uvec>(find(
      params.Get<arma::Mat<size_t>>("training_labels") == 2)).n_rows == 47);

  REQUIRE(static_cast<uvec>(find(
      params.Get<arma::Mat<size_t>>("test_labels") == 0)).n_rows == 12);
  REQUIRE(static_cast<uvec>(find(
      params.Get<arma::Mat<size_t>>("test_labels") == 1)).n_rows == 30);
  REQUIRE(static_cast<uvec>(find(
      params.Get<arma::Mat<size_t>>("test_labels") == 2)).n_rows == 20);
}
