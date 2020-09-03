/**
 * @file tests/one_hot_encoding_test.cpp
 * @author Jeffin Sam
 *
 * Tests for the One-Hot Encoding method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "test_catch_tools.hpp"
#include "catch.hpp"
#include <mlpack/core/data/one_hot_encoding.hpp>

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

/**
 * Test one hot encoding for small input.
 */
TEST_CASE("OneHotEncodingTest01", "[OneHotEncodingTest]")
{
  arma::Mat<size_t> matrix;
  matrix = "1 0;"
           "0 1;"
           "1 0;"
           "1 0;"
           "1 0;"
           "1 0;"
           "0 1;"
           "1 0;";
  // Output matrix to save one-hot encoding results.
  arma::Mat<size_t> output;
  matrix = matrix.t();
  arma::irowvec labels("-1 1 -1 -1 -1 -1 1 -1");
  data::OneHotEncoding(labels, output);

  REQUIRE(matrix.n_cols == output.n_cols);
  REQUIRE(matrix.n_rows == output.n_rows);
  CheckMatrices(output, matrix);
}

/**
 * Test one hot encoding for sparse matrix.
 */
TEST_CASE("OneHotEncodingSparseMatTest", "[OneHotEncodingTest]")
{
  arma::SpMat<size_t> matrix;
  matrix = "1 0;"
           "0 1;"
           "1 0;"
           "1 0;"
           "1 0;"
           "1 0;"
           "0 1;"
           "1 0;";

  matrix = matrix.t();
  // Output matrix to save one-hot encoding  results.
  arma::SpMat<size_t> output;
  arma::irowvec labels("-1 1 -1 -1 -1 -1 1 -1");
  data::OneHotEncoding(labels, output);

  REQUIRE(matrix.n_cols == output.n_cols);
  REQUIRE(matrix.n_rows == output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    REQUIRE(matrix.at(i) == output.at(i));
}

/**
 * Test one hot encoding for a particular row.
 */
TEST_CASE("OneHotEncodingInputTest", "[OneHotEncodingTest]")
{
  arma::Mat<int> matrix;
  arma::Mat<int> input;
  input = "1 1 -1 -1 -1 -1 1 1;"
           "-1 1 -1 -1 -1 -1 1 -1;";
  matrix = "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;";

  // Output matrix to save one-hot encoding results.
  arma::Mat<int> output;
  arma::Col<size_t> indices("1");
  data::OneHotEncoding(input, indices, output);

  REQUIRE(matrix.n_cols == output.n_cols);
  REQUIRE(matrix.n_rows == output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    REQUIRE(matrix.at(i) == output.at(i));
}

/**
 * Test one hot encoding for big input.
 */
TEST_CASE("OneHotEncodingBigInputTest", "[OneHotEncodingTest]")
{
  arma::Mat<int> matrix;
  arma::Mat<int> input;
  input = "1 1 -1 -1 -1 -1 1 1;"
          "-1 1 -1 -1 -1 -1 1 -1;"
          "1 1 -1 -1 -1 -1 1 1;"
          "-1 1 -1 -1 -1 -1 1 -1;"
          "1 1 -1 -1 -1 -1 1 1;";

  matrix = "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;";

  // Output matrix to save one-hot encoding results.
  arma::Mat<int> output;
  arma::Col<size_t> indices("1 3");
  data::OneHotEncoding(input, indices, output);

  REQUIRE(matrix.n_cols == output.n_cols);
  REQUIRE(matrix.n_rows == output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    REQUIRE(matrix.at(i) == output.at(i));
}

/**
 * Test one hot encoding for very big input.
 */
TEST_CASE("OneHotEncodingVeryBigInputTest", "[OneHotEncodingTest]")
{
  arma::Mat<int> matrix;
  arma::Mat<int> input;
  input = "1 1 -1 -1 -1 -1 1 1;"
          "-1 1 -1 -1 -1 -1 1 -1;"
          "1 1 -1 -1 -1 -1 1 1;"
          "1 1 -1 -1 -1 -1 1 1;"
          "-1 1 -1 -1 -1 -1 1 -1;"
          "1 1 -1 -1 -1 -1 1 1;"
          "1 1 -1 -1 -1 -1 1 1;"
          "-1 1 -1 -1 -1 -1 1 -1;"
          "-1 1 -1 -1 -1 -1 1 -1;"
          "1 1 -1 -1 -1 -1 1 1;"
          "1 1 -1 -1 -1 -1 1 1;";

  matrix = "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 1 -1 -1 -1 -1 1 1;";

  // Output matrix to save one-hot encoding results.
  arma::Mat<int> output;
  arma::Col<size_t> indices("1 4 7 8");
  data::OneHotEncoding(input, indices, output);

  REQUIRE(matrix.n_cols == output.n_cols);
  REQUIRE(matrix.n_rows == output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    REQUIRE(matrix.at(i) == output.at(i));
}

/**
 * Test one hot encoding using DatasetInfo object.
 */
TEST_CASE("OneHotEncodingDatasetinfoTest", "[OneHotEncodingTest]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, hello" << endl;
  f << "3, 4, goodbye" << endl;
  f << "5, 6, coffee" << endl;
  f << "7, 8, confusion" << endl;
  f << "9, 10, hello" << endl;
  f << "11, 12, confusion" << endl;
  f << "13, 14, confusion" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  DatasetInfo info;
  data::Load("test.csv", matrix, info);
  arma::umat output;
  data::OneHotEncoding(matrix, output, info);
  REQUIRE(output.n_cols == 7);
  REQUIRE(output.n_rows == 6);
  REQUIRE(info.Type(0) == Datatype::numeric);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::categorical);

  REQUIRE(output(0, 0) == 1);
  REQUIRE(output(1, 0) == 2);
  REQUIRE(output(2, 0) == 1);
  REQUIRE(output(0, 1) == 3);
  REQUIRE(output(1, 1) == 4);
  REQUIRE(output(3, 1) == 1);
  REQUIRE(output(0, 2) == 5);
  REQUIRE(output(1, 2) == 6);
  REQUIRE(output(4, 2) == 1);
  REQUIRE(output(0, 3) == 7);
  REQUIRE(output(1, 3) == 8);
  REQUIRE(output(5, 3) == 1);
  REQUIRE(output(0, 4) == 9);
  REQUIRE(output(1, 4) == 10);
  REQUIRE(output(2, 4) == 1);
  REQUIRE(output(0, 5) == 11);
  REQUIRE(output(1, 5) == 12);
  REQUIRE(output(5, 5) == 1);
  REQUIRE(output(0, 6) == 13);
  REQUIRE(output(1, 6) == 14);
  REQUIRE(output(5, 6) == 1);
  REQUIRE(output(3, 0) == 0);
  REQUIRE(output(4, 0) == 0);
  REQUIRE(output(3, 0) == 0);
  REQUIRE(output(4, 0) == 0);
  REQUIRE(output(5, 0) == 0);

  remove("test.csv");
}
