/**
 * @file one_hot_encoding_test.cpp
 * @author Jeffin Sam
 *
 * Tests for onehotencoding().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <sstream>

#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

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
// Output matrix to save onehotencoding results.
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
  arma::ucolvec indices("1");
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

  // Output matrix to save onehotencoding results.
  arma::Mat<int> output;
  arma::ucolvec indices("1 3");
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

// Output matrix to save onehotencoding results.
  arma::Mat<int> output;
  arma::ucolvec indices("1 4 7 8");
  data::OneHotEncoding(input, indices, output);

  BOOST_REQUIRE_EQUAL(matrix.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, output.n_rows);
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_EQUAL(matrix.at(i), output.at(i));
}

BOOST_AUTO_TEST_SUITE_END();
