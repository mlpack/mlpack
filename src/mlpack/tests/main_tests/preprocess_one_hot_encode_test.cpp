/**
 * @file tests/main_tests/preprocess_one_hot_encode_test.cpp
 * @author Jeffin Sam
 *
 * Test RUN_BINDING() of preprocess_one_hot_encoding_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/preprocess/preprocess_one_hot_encoding_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(PreprocessOneHotEncodingTestFixture);

/**
 * Test one hot encoding binding.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "PreprocessOneHotEncodingTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  arma::mat matrix;
  matrix = "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {1, 3});
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(matrix.n_cols == output.n_cols);
  REQUIRE(matrix.n_rows == output.n_rows);
  CheckMatrices(output, matrix);
}

/**
 * Test for empty matrix.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "EmptyMatrixTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset;

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {1, 3});
  // This will throw an error since dimensions are bigger than the matrix.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Test empty vector as input for dimensions.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "EmptyIndicesTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {});
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(dataset.n_cols == output.n_cols);
  REQUIRE(dataset.n_rows == output.n_rows);
  CheckMatrices(output, dataset);
}

/**
 * Test for invalid dimension, larger than count of rows.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "InvalidDimensionTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {10000});
  // Error since dimensions are bigger than matrix.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Test for negative dimensions.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "NegativeDimensionTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {-10000});
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Test for empty matrix and empty dimensions vector.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "EmptyMatrixEmptyIndicesTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset;

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {});
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(dataset.n_cols == output.n_cols);
  REQUIRE(dataset.n_rows == output.n_rows);
  CheckMatrices(output, dataset);
}
