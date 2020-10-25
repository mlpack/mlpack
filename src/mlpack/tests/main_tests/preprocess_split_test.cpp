/**
 * @file tests/main_tests/preprocess_split_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of preprocess_split_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "PreprocessSplit";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_split_main.cpp>

#include "test_helper.hpp"
#include "../test_catch_tools.hpp"
#include "../catch.hpp"

#include <cmath>

using namespace mlpack;

struct PreprocessSplitTestFixture
{
 public:
  PreprocessSplitTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~PreprocessSplitTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

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
  data::Load("vc2.csv", inputData);
  data::Load("vc2_labels.txt", labels);

  // Store size of input dataset.
  int inputSize  = inputData.n_cols;
  int labelSize  = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  // Input test_ratio.
  SetInputParam("test_ratio", (double) 0.1);

  mlpackMain();

  // Now check that the output has desired dimensions.
  REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
      std::ceil(0.9 * inputSize));
  REQUIRE(IO::GetParam<arma::mat>("test").n_cols ==
      std::floor(0.1 * inputSize));

  REQUIRE(
      IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols ==
      std::ceil(0.9 * labelSize));
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols ==
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
  data::Load("vc2.csv", inputData);

  // Store size of input dataset.
  int inputSize  = inputData.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));

  // Input test_ratio.
  SetInputParam("test_ratio", (double) 0.1);

  mlpackMain();

  // Now check that the output has desired dimensions.
  REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
      std::ceil(0.9 * inputSize));
  REQUIRE(IO::GetParam<arma::mat>("test").n_cols ==
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
  data::Load("vc2.csv", inputData);
  data::Load("vc2_labels.txt", labels);

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) -0.2);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  data::Load("vc2.csv", inputData);
  data::Load("vc2_labels.txt", labels);

  // Store size of input dataset.
  int inputSize = inputData.n_cols;
  int labelSize = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) 0.0);

  mlpackMain();

  // Now check that the output has desired dimensions.
  REQUIRE(IO::GetParam<arma::mat>("training").n_cols == inputSize);
  REQUIRE(IO::GetParam<arma::mat>("test").n_cols == 0);

  REQUIRE(IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols ==
      labelSize);
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols == 0);
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
  data::Load("vc2.csv", inputData);
  data::Load("vc2_labels.txt", labels);

  // Store size of input dataset.
  int inputSize = inputData.n_cols;
  int labelSize = labels.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("input_labels", std::move(labels));

  SetInputParam("test_ratio", (double) 1.0);

  mlpackMain();

  // Now check that the output has desired dimensions.
  REQUIRE(IO::GetParam<arma::mat>("training").n_cols == 0);
  REQUIRE(IO::GetParam<arma::mat>("test").n_cols == inputSize);

  REQUIRE(IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols == 0);
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols == labelSize);
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
  data::Load("vc2.csv", inputData);

  // Store size of input dataset.
  int inputSize  = inputData.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", inputData);

  // Input test_ratio.
  SetInputParam("test_ratio", (double) 0.1);
  SetInputParam("no_shuffle", true);
  mlpackMain();

  // Now check that the output has desired dimensions.
  REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
      std::ceil(0.9 * inputSize));
  REQUIRE(IO::GetParam<arma::mat>("test").n_cols ==
      std::floor(0.1 * inputSize));

  arma::mat concat = arma::join_rows(IO::GetParam<arma::mat>("training"),
      IO::GetParam<arma::mat>("test"));
  CheckMatrices(inputData, concat);
}
