/**
 * @file tests/arma_extend_test.cpp
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
#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace arma;


/**
 * Test const_row_col_iterator for basic functionality.
 */
TEST_CASE("ConstRowColIteratorTest", "[ArmaExtendTest]")
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
  for (it = X.begin_row_col(); it != X.end_row_col(); ++it)
  {
    // Check iterator value.
    REQUIRE(*it == ((count % 5) * 3 + (count / 5)));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);

    ++count;
  }
  REQUIRE(count == 25);
  it = X.end_row_col();
  do
  {
    --it;
    --count;

    // Check iterator value.
    REQUIRE(*it == ((count % 5) * 3 + (count / 5)));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);
  } while (it != X.begin_row_col());

  REQUIRE(count == 0);
}

/**
 * Test row_col_iterator for basic functionality.
 */
TEST_CASE("RowColIteratorTest", "[ArmaExtendTest]")
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
  for (it = X.begin_row_col(); it != X.end_row_col(); ++it)
  {
    // Check iterator value.
    REQUIRE(*it == ((count % 5) * 3 + (count / 5)));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);

    ++count;
  }
  REQUIRE(count == 25);
  it = X.end_row_col();
  do
  {
    --it;
    --count;

    // Check iterator value.
    REQUIRE(*it == ((count % 5) * 3 + (count / 5)));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);
  } while (it != X.begin_row_col());

  REQUIRE(count == 0);
}

/**
 * Operator-- test for mat::row_col_iterator and mat::const_row_col_iterator
 */
TEST_CASE("MatRowColIteratorDecrementOperatorTest", "[ArmaExtendTest]")
{
  mat test = ones<mat>(5, 5);

  mat::row_col_iterator it1 = test.begin_row_col();
  mat::row_col_iterator it2 = it1;

  // Check that postfix-- does not decrement the position when position is
  // pointing to the beginning.
  auto junk = it2--; (void)(junk);
  REQUIRE(it1.row() == it2.row());
  REQUIRE(it1.col() == it2.col());

  // Check that prefix-- does not decrement the position when position is
  // pointing to the beginning.
  --it2;
  REQUIRE(it1.row() == it2.row());
  REQUIRE(it1.col() == it2.col());
}

// These tests don't work when the sparse iterators hold references and not
// pointers internally because of the lack of default constructor.

/**
 * Test sparse const_row_col_iterator for basic functionality.
 */
TEST_CASE("ConstSpRowColIteratorTest", "[ArmaExtendTest]")
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
  for (it = X.begin_row_col(); it != X.end_row_col(); ++it)
  {
    // Check iterator value.
    REQUIRE(*it == (count % 5) * 3 + (count / 5));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);

    ++count;
  }
  REQUIRE(count == 25);
  it = X.end_row_col();
  do
  {
    --it;
    --count;

    // Check iterator value.
    REQUIRE(*it == ((count % 5) * 3 + (count / 5)));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);
  } while (it != X.begin_row_col());

  REQUIRE(count == 1);
}

/**
 * Test sparse row_col_iterator for basic functionality.
 */
TEST_CASE("SpRowColIteratorTest", "[ArmaExtendTest]")
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
  for (it = X.begin_row_col(); it != X.end_row_col(); ++it)
  {
    // Check iterator value.
    REQUIRE(*it == ((count % 5) * 3 + (count / 5)));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);

    ++count;
  }
  REQUIRE(count == 25);
  it = X.end_row_col();
  do
  {
    --it;
    --count;

    // Check iterator value.
    REQUIRE(*it == ((count % 5) * 3 + (count / 5)));

    // Check iterator position.
    REQUIRE(it.row() == count % 5);
    REQUIRE(it.col() == count / 5);
  } while (it != X.begin_row_col());

  REQUIRE(count == 1);
}
