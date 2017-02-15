/**
 * @file arma_extend_test.cpp
 * @author Ryan Curtin
 *
 * Test of the mlpack extensions to Armadillo.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace arma;

BOOST_AUTO_TEST_SUITE(ArmaExtendTest);

/**
 * Make sure we can reshape a matrix in-place without changing anything.
 */
BOOST_AUTO_TEST_CASE(InplaceReshapeColumnTest)
{
  arma::mat X;
  X.randu(1, 10);
  arma::mat oldX = X;

  arma::inplace_reshape(X, 2, 5);

  BOOST_REQUIRE_EQUAL(X.n_rows, 2);
  BOOST_REQUIRE_EQUAL(X.n_cols, 5);
  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_CLOSE(X[i], oldX[i], 1e-5); // Order should be preserved.
}

/**
 * Make sure we can reshape a large matrix.
 */
BOOST_AUTO_TEST_CASE(InplaceReshapeMatrixTest)
{
  arma::mat X;
  X.randu(8, 10);
  arma::mat oldX = X;

  arma::inplace_reshape(X, 10, 8);

  BOOST_REQUIRE_EQUAL(X.n_rows, 10);
  BOOST_REQUIRE_EQUAL(X.n_cols, 8);
  for (size_t i = 0; i < 80; ++i)
    BOOST_REQUIRE_CLOSE(X[i], oldX[i], 1e-5); // Order should be preserved.
}

/**
 * Test const_row_col_iterator for basic functionality.
 */
BOOST_AUTO_TEST_CASE(ConstRowColIteratorTest)
{
  mat X;
  X.zeros(5, 5);
  for (size_t i = 0; i < 5; ++i)
    X.col(i) += i;

  for (size_t i = 0; i < 5; ++i)
    X.row(i) += 3 * i;

  // Make sure default constructor works okay.
  mat::const_row_col_iterator it;
  // Make sure ++ operator, operator* and comparison operators work fine.
  size_t count = 0;
  for (it = X.begin_row_col(); it != X.end_row_col(); it++)
  {
    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);

    count++;
  }
  BOOST_REQUIRE_EQUAL(count, 25);
  it = X.end_row_col();
  do
  {
    it--;
    count--;

    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);
  } while (it != X.begin_row_col());

  BOOST_REQUIRE_EQUAL(count, 0);
}

/**
 * Test row_col_iterator for basic functionality.
 */
BOOST_AUTO_TEST_CASE(RowColIteratorTest)
{
  mat X;
  X.zeros(5, 5);
  for (size_t i = 0; i < 5; ++i)
    X.col(i) += i;

  for (size_t i = 0; i < 5; ++i)
    X.row(i) += 3 * i;

  // Make sure default constructor works okay.
  mat::row_col_iterator it;
  // Make sure ++ operator, operator* and comparison operators work fine.
  size_t count = 0;
  for (it = X.begin_row_col(); it != X.end_row_col(); it++)
  {
    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);

    count++;
  }
  BOOST_REQUIRE_EQUAL(count, 25);
  it = X.end_row_col();
  do
  {
    it--;
    count--;

    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);
  } while (it != X.begin_row_col());

  BOOST_REQUIRE_EQUAL(count, 0);
}

/**
 * Operator-- test for mat::row_col_iterator and mat::const_row_col_iterator
 */
BOOST_AUTO_TEST_CASE(MatRowColIteratorDecrementOperatorTest)
{
  mat test = ones<mat>(5, 5);

  mat::row_col_iterator it1 = test.begin_row_col();
  mat::row_col_iterator it2 = it1;

  // check that postfix-- does not decrement the position when position is
  // pointing to the begining
  it2--;
  BOOST_REQUIRE_EQUAL(it1.row(), it2.row());
  BOOST_REQUIRE_EQUAL(it1.col(), it2.col());

  // check that prefix-- does not decrement the position when position is
  // pointing to the begining
  --it2;
  BOOST_REQUIRE_EQUAL(it1.row(), it2.row());
  BOOST_REQUIRE_EQUAL(it1.col(), it2.col());
}

// These tests don't work when the sparse iterators hold references and not
// pointers internally because of the lack of default constructor.
#if ARMA_VERSION_MAJOR > 4 || \
    (ARMA_VERSION_MAJOR == 4 && ARMA_VERSION_MINOR > 320)

/**
 * Test sparse const_row_col_iterator for basic functionality.
 */
BOOST_AUTO_TEST_CASE(ConstSpRowColIteratorTest)
{
  sp_mat X(5, 5);
  for (size_t i = 0; i < 5; ++i)
    X.col(i) += i;

  for (size_t i = 0; i < 5; ++i)
    X.row(i) += 3 * i;

  // Make sure default constructor works okay.
  sp_mat::const_row_col_iterator it;
  // Make sure ++ operator, operator* and comparison operators work fine.
  size_t count = 1;
  for (it = X.begin_row_col(); it != X.end_row_col(); it++)
  {
    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);

    count++;
  }
  BOOST_REQUIRE_EQUAL(count, 25);
  it = X.end_row_col();
  do
  {
    it--;
    count--;

    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);
  } while (it != X.begin_row_col());

  BOOST_REQUIRE_EQUAL(count, 1);
}

/**
 * Test sparse row_col_iterator for basic functionality.
 */
BOOST_AUTO_TEST_CASE(SpRowColIteratorTest)
{
  sp_mat X(5, 5);
  for (size_t i = 0; i < 5; ++i)
    X.col(i) += i;

  for (size_t i = 0; i < 5; ++i)
    X.row(i) += 3 * i;

  // Make sure default constructor works okay.
  sp_mat::row_col_iterator it;
  // Make sure ++ operator, operator* and comparison operators work fine.
  size_t count = 1;
  for (it = X.begin_row_col(); it != X.end_row_col(); it++)
  {
    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);

    count++;
  }
  BOOST_REQUIRE_EQUAL(count, 25);
  it = X.end_row_col();
  do
  {
    it--;
    count--;

    // Check iterator value.
    BOOST_REQUIRE_EQUAL(*it, (count % 5) * 3 + (count / 5));

    // Check iterator position.
    BOOST_REQUIRE_EQUAL(it.row(), count % 5);
    BOOST_REQUIRE_EQUAL(it.col(), count / 5);
  } while (it != X.begin_row_col());

  BOOST_REQUIRE_EQUAL(count, 1);
}

#endif

BOOST_AUTO_TEST_SUITE_END();
