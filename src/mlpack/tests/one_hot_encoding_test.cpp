/**
 * @file one_hot_encoding_test.cpp
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
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include <mlpack/core/data/one_hot_encoding.hpp>

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(OneHotEncoding);

/**
 * Test one hot encoding.
 */
BOOST_AUTO_TEST_CASE(OneHotEncodingTest)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, output.n_rows);
  CheckMatrices(output, matrix);
}

BOOST_AUTO_TEST_CASE(OneHotEncodingSparseMatTest)
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
  // Output matrix to save onehotencoding results.
  arma::SpMat<size_t> output;
  arma::irowvec labels("-1 1 -1 -1 -1 -1 1 -1");
  data::OneHotEncoding(labels, output);

  BOOST_REQUIRE_EQUAL(matrix.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_EQUAL(matrix.at(i), output.at(i));
}

BOOST_AUTO_TEST_CASE(OneHotEncodingInputTest)
{
  arma::Mat<int> matrix;
  arma::Mat<int> input;
  input = "1 1 -1 -1 -1 -1 1 1;"
           "-1 1 -1 -1 -1 -1 1 -1;";
  matrix = "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;";

  // Output matrix to save onehotencoding results.
  arma::Mat<int> output;
  arma::Col<size_t> indices("1");
  data::OneHotEncoding(input, indices, output);

  BOOST_REQUIRE_EQUAL(matrix.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_EQUAL(matrix.at(i), output.at(i));
}

BOOST_AUTO_TEST_CASE(OneHotEncodingBigInputTest)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_EQUAL(matrix.at(i), output.at(i));
}

BOOST_AUTO_TEST_CASE(OneHotEncodingVeryBigInputTest)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_EQUAL(matrix.at(i), output.at(i));
}

BOOST_AUTO_TEST_CASE(OneHotEncodingDatasetinfoTest)
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
  BOOST_REQUIRE_EQUAL(output.n_cols, 7);
  BOOST_REQUIRE_EQUAL(output.n_rows, 6);
  BOOST_REQUIRE(info.Type(0) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::categorical);

  BOOST_REQUIRE_EQUAL(output(0, 0), 1);
  BOOST_REQUIRE_EQUAL(output(1, 0), 2);
  BOOST_REQUIRE_EQUAL(output(2, 0), 1);
  BOOST_REQUIRE_EQUAL(output(0, 1), 3);
  BOOST_REQUIRE_EQUAL(output(1, 1), 4);
  BOOST_REQUIRE_EQUAL(output(3, 1), 1);
  BOOST_REQUIRE_EQUAL(output(0, 2), 5);
  BOOST_REQUIRE_EQUAL(output(1, 2), 6);
  BOOST_REQUIRE_EQUAL(output(4, 2), 1);
  BOOST_REQUIRE_EQUAL(output(0, 3), 7);
  BOOST_REQUIRE_EQUAL(output(1, 3), 8);
  BOOST_REQUIRE_EQUAL(output(5, 3), 1);
  BOOST_REQUIRE_EQUAL(output(0, 4), 9);
  BOOST_REQUIRE_EQUAL(output(1, 4), 10);
  BOOST_REQUIRE_EQUAL(output(2, 4), 1);
  BOOST_REQUIRE_EQUAL(output(0, 5), 11);
  BOOST_REQUIRE_EQUAL(output(1, 5), 12);
  BOOST_REQUIRE_EQUAL(output(5, 5), 1);
  BOOST_REQUIRE_EQUAL(output(0, 6), 13);
  BOOST_REQUIRE_EQUAL(output(1, 6), 14);
  BOOST_REQUIRE_EQUAL(output(5, 6), 1);

  remove("test.csv");
}
BOOST_AUTO_TEST_SUITE_END();
