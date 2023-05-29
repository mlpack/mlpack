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

  data::DatasetInfo di(dataset.n_rows);
  SetInputParam("input", std::make_tuple(di, dataset));
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

  data::DatasetInfo di(dataset.n_rows);
  SetInputParam("input", std::make_tuple(di, dataset));
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

  data::DatasetInfo di(dataset.n_rows);
  SetInputParam("input", std::make_tuple(di, dataset));
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

  data::DatasetInfo di(dataset.n_rows);
  SetInputParam("input", std::make_tuple(di, dataset));
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

  data::DatasetInfo di(dataset.n_rows);
  SetInputParam("input", std::make_tuple(di, dataset));
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
  data::DatasetInfo di(dataset.n_rows);

  SetInputParam("input", std::make_tuple(di, dataset));
  SetInputParam<vector<int>>("dimensions", {});
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(dataset.n_cols == output.n_cols);
  REQUIRE(dataset.n_rows == output.n_rows);
  CheckMatrices(output, dataset);
}

/**
 * Test for a dataset with categorical features, where we one-hot encode all
 * categorical features.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "CategoricalMatrixTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset(5, 4);
  dataset.randu();

  // Dimension 2 will be categorical.
  dataset(0, 2) = 0;
  dataset(1, 2) = 1;
  dataset(2, 2) = 1;
  dataset(3, 2) = 2;
  dataset(4, 2) = 0;

  data::DatasetInfo info(4);
  info.Type(2) = data::Datatype::categorical;

  SetInputParam("input", std::make_tuple(info, dataset));
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(dataset.n_cols == output.n_cols);
  REQUIRE(dataset.n_rows + 2 == output.n_cols);

  // Make sure one-hot encoding was correct.
  REQUIRE(output(0, 2) == 1);
  REQUIRE(output(1, 2) == 0);
  REQUIRE(output(2, 2) == 0);
  REQUIRE(output(3, 2) == 0);
  REQUIRE(output(4, 2) == 1);
  REQUIRE(output(0, 3) == 0);
  REQUIRE(output(1, 3) == 1);
  REQUIRE(output(2, 3) == 1);
  REQUIRE(output(3, 3) == 0);
  REQUIRE(output(4, 3) == 0);
  REQUIRE(output(0, 4) == 0);
  REQUIRE(output(1, 4) == 0);
  REQUIRE(output(2, 4) == 0);
  REQUIRE(output(3, 4) == 1);
  REQUIRE(output(4, 4) == 0);
}

/**
 * Test for a dataset with no categorical features, where we don't specify the
 * dimensions to convert.  This should convert nothing.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "NoCategoricalMatrixTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset(5, 4);
  dataset.randu();
  data::DatasetInfo info(4); // all numeric dimensions

  SetInputParam("input", std::make_tuple(info, dataset));
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(dataset.n_cols == output.n_cols);
  REQUIRE(dataset.n_rows == output.n_cols);
  CheckMatrices(output, dataset);
}

/**
 * Test for a dataset with multiple categorical features.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture, "MultipleFeatureCategoricalMatrixTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset(5, 4);
  dataset.randu();

  // Dimensions 0, 2, and 3 will be categorical.
  dataset(0, 0) = 0;
  dataset(1, 0) = 1;
  dataset(2, 0) = 2;
  dataset(3, 0) = 3;
  dataset(4, 0) = 3;
  dataset(0, 2) = 0;
  dataset(1, 2) = 1;
  dataset(2, 2) = 1;
  dataset(3, 2) = 2;
  dataset(4, 2) = 0;
  dataset(0, 3) = 0;
  dataset(1, 3) = 0;
  dataset(2, 3) = 1;
  dataset(3, 3) = 1;
  dataset(4, 3) = 1;

  data::DatasetInfo info(4);
  info.Type(0) = data::Datatype::categorical;
  info.Type(2) = data::Datatype::categorical;
  info.Type(3) = data::Datatype::categorical;

  SetInputParam("input", std::make_tuple(info, dataset));
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(dataset.n_cols == output.n_cols);
  REQUIRE(dataset.n_rows + 3 + 2 + 1 == output.n_cols);

  // Make sure one-hot encoding was correct.
  REQUIRE(output(0, 0) == 1);
  REQUIRE(output(1, 0) == 0);
  REQUIRE(output(2, 0) == 0);
  REQUIRE(output(3, 0) == 0);
  REQUIRE(output(4, 0) == 0);
  REQUIRE(output(0, 1) == 0);
  REQUIRE(output(1, 1) == 1);
  REQUIRE(output(2, 1) == 0);
  REQUIRE(output(3, 1) == 0);
  REQUIRE(output(4, 1) == 0);
  REQUIRE(output(0, 2) == 0);
  REQUIRE(output(1, 2) == 0);
  REQUIRE(output(2, 2) == 1);
  REQUIRE(output(3, 2) == 0);
  REQUIRE(output(4, 2) == 0);
  REQUIRE(output(0, 3) == 0);
  REQUIRE(output(1, 3) == 0);
  REQUIRE(output(2, 3) == 0);
  REQUIRE(output(3, 3) == 1);
  REQUIRE(output(4, 3) == 1);

  REQUIRE(output(0, 5) == 1);
  REQUIRE(output(1, 5) == 0);
  REQUIRE(output(2, 5) == 0);
  REQUIRE(output(3, 5) == 0);
  REQUIRE(output(4, 5) == 1);
  REQUIRE(output(0, 6) == 0);
  REQUIRE(output(1, 6) == 1);
  REQUIRE(output(2, 6) == 1);
  REQUIRE(output(3, 6) == 0);
  REQUIRE(output(4, 6) == 0);
  REQUIRE(output(0, 7) == 0);
  REQUIRE(output(1, 7) == 0);
  REQUIRE(output(2, 7) == 0);
  REQUIRE(output(3, 7) == 1);
  REQUIRE(output(4, 7) == 0);

  REQUIRE(output(0, 8) == 1);
  REQUIRE(output(1, 8) == 1);
  REQUIRE(output(2, 8) == 0);
  REQUIRE(output(3, 8) == 0);
  REQUIRE(output(4, 8) == 0);
  REQUIRE(output(0, 9) == 0);
  REQUIRE(output(1, 9) == 0);
  REQUIRE(output(2, 9) == 1);
  REQUIRE(output(3, 9) == 1);
  REQUIRE(output(4, 9) == 1);
}

/**
 * Test for a dataset with multiple categorical features, where we are not
 * converting them all.
 */
TEST_CASE_METHOD(
    PreprocessOneHotEncodingTestFixture,
    "MultipleNotAllFeatureCategoricalMatrixTest",
    "[PreprocessOneHotEncodingMainTest][BindingTests]")
{
  arma::mat dataset(5, 4);
  dataset.randu();

  // Dimensions 0, 2, and 3 will be categorical, but we will only convert
  // dimensions 0 and 2.
  dataset(0, 0) = 0;
  dataset(1, 0) = 1;
  dataset(2, 0) = 2;
  dataset(3, 0) = 3;
  dataset(4, 0) = 3;
  dataset(0, 2) = 0;
  dataset(1, 2) = 1;
  dataset(2, 2) = 1;
  dataset(3, 2) = 2;
  dataset(4, 2) = 0;
  dataset(0, 3) = 0;
  dataset(1, 3) = 0;
  dataset(2, 3) = 1;
  dataset(3, 3) = 1;
  dataset(4, 3) = 1;

  data::DatasetInfo info(4);
  info.Type(0) = data::Datatype::categorical;
  info.Type(2) = data::Datatype::categorical;
  info.Type(3) = data::Datatype::categorical;

  SetInputParam("input", std::make_tuple(info, dataset));
  SetInputParam<vector<int>>("dimensions", {0, 2});
  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(dataset.n_cols == output.n_cols);
  REQUIRE(dataset.n_rows + 3 + 2 == output.n_cols);

  // Make sure one-hot encoding was correct.
  REQUIRE(output(0, 0) == 1);
  REQUIRE(output(1, 0) == 0);
  REQUIRE(output(2, 0) == 0);
  REQUIRE(output(3, 0) == 0);
  REQUIRE(output(4, 0) == 0);
  REQUIRE(output(0, 1) == 0);
  REQUIRE(output(1, 1) == 1);
  REQUIRE(output(2, 1) == 0);
  REQUIRE(output(3, 1) == 0);
  REQUIRE(output(4, 1) == 0);
  REQUIRE(output(0, 2) == 0);
  REQUIRE(output(1, 2) == 0);
  REQUIRE(output(2, 2) == 1);
  REQUIRE(output(3, 2) == 0);
  REQUIRE(output(4, 2) == 0);
  REQUIRE(output(0, 3) == 0);
  REQUIRE(output(1, 3) == 0);
  REQUIRE(output(2, 3) == 0);
  REQUIRE(output(3, 3) == 1);
  REQUIRE(output(4, 3) == 1);

  REQUIRE(output(0, 5) == 1);
  REQUIRE(output(1, 5) == 0);
  REQUIRE(output(2, 5) == 0);
  REQUIRE(output(3, 5) == 0);
  REQUIRE(output(4, 5) == 1);
  REQUIRE(output(0, 6) == 0);
  REQUIRE(output(1, 6) == 1);
  REQUIRE(output(2, 6) == 1);
  REQUIRE(output(3, 6) == 0);
  REQUIRE(output(4, 6) == 0);
  REQUIRE(output(0, 7) == 0);
  REQUIRE(output(1, 7) == 0);
  REQUIRE(output(2, 7) == 0);
  REQUIRE(output(3, 7) == 1);
  REQUIRE(output(4, 7) == 0);

  // Make sure we did not one-hot encode the last dimension.
  REQUIRE(output(0, 8) == 0);
  REQUIRE(output(1, 8) == 0);
  REQUIRE(output(2, 8) == 1);
  REQUIRE(output(3, 8) == 1);
  REQUIRE(output(4, 8) == 1);
}
