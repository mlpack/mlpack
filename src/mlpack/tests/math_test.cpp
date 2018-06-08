/**
 * @file math_test.cpp
 * @author Ryan Curtin
 *
 * Tests for everything in the math:: namespace.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/math/clamp.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/math/range.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace math;

BOOST_AUTO_TEST_SUITE(MathTest);

/**
 * Verify that the empty constructor creates an empty range.
 */
BOOST_AUTO_TEST_CASE(RangeEmptyConstructor)
{
  Range x = Range();

  // Just verify that it is empty.
  BOOST_REQUIRE_GT(x.Lo(), x.Hi());
}

/**
 * Verify that the point constructor correctly creates a range that is just a
 * point.
 */
BOOST_AUTO_TEST_CASE(RangePointConstructor)
{
  Range x(10.0);

  BOOST_REQUIRE_CLOSE(x.Lo(), x.Hi(), 1e-25);
  BOOST_REQUIRE_SMALL(x.Width(), 1e-5);
  BOOST_REQUIRE_CLOSE(x.Lo(), 10.0, 1e-25);
  BOOST_REQUIRE_CLOSE(x.Hi(), 10.0, 1e-25);
}

/**
 * Verify that the range constructor correctly creates the range.
 */
BOOST_AUTO_TEST_CASE(RangeConstructor)
{
  Range x(0.5, 5.5);

  BOOST_REQUIRE_CLOSE(x.Lo(), 0.5, 1e-25);
  BOOST_REQUIRE_CLOSE(x.Hi(), 5.5, 1e-25);
}

/**
 * Test that we get the width correct.
 */
BOOST_AUTO_TEST_CASE(RangeWidth)
{
  Range x(0.0, 10.0);

  BOOST_REQUIRE_CLOSE(x.Width(), 10.0, 1e-20);

  // Make it empty.
  x.Hi() = 0.0;

  BOOST_REQUIRE_SMALL(x.Width(), 1e-5);

  // Make it negative.
  x.Hi() = -2.0;

  BOOST_REQUIRE_SMALL(x.Width(), 1e-5);

  // Just one more test.
  x.Lo() = -5.2;
  x.Hi() = 5.2;

  BOOST_REQUIRE_CLOSE(x.Width(), 10.4, 1e-5);
}

/**
 * Test that we get the midpoint correct.
 */
BOOST_AUTO_TEST_CASE(RangeMidpoint)
{
  Range x(0.0, 10.0);

  BOOST_REQUIRE_CLOSE(x.Mid(), 5.0, 1e-5);

  x.Lo() = -5.0;

  BOOST_REQUIRE_CLOSE(x.Mid(), 2.5, 1e-5);
}

/**
 * Test that we can expand to include other ranges correctly.
 */
BOOST_AUTO_TEST_CASE(RangeIncludeOther)
{
  // We need to test both |= and |.
  // We have three cases: non-overlapping; overlapping; equivalent, and then a
  // couple permutations (switch left with right and make sure it still works).
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  Range z(0.0, 2.0); // Used for operator|=().
  Range w;
  z |= y;
  w = x | y;

  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 5.0, 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 5.0, 1e-5);

  // Switch operator precedence.
  z = y;
  z |= x;
  w = y | x;

  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 5.0, 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 5.0, 1e-5);

  // Now make them overlapping.
  x = Range(0.0, 3.5);
  y = Range(3.0, 4.0);

  z = x;
  z |= y;
  w = x | y;

  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 4.0, 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 4.0, 1e-5);

  // Switch operator precedence.
  z = y;
  z |= x;
  w = y | x;

  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 4.0, 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 4.0, 1e-5);

  // Now the equivalent case.
  x = Range(0.0, 2.0);
  y = Range(0.0, 2.0);

  z = x;
  z |= y;
  w = x | y;

  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 2.0, 1e-5);

  z = y;
  z |= x;
  w = y | x;

  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 2.0, 1e-5);
}

/**
 * Test that we can 'and' ranges correctly.
 */
BOOST_AUTO_TEST_CASE(RangeIntersectOther)
{
  // We need to test both &= and &.
  // We have three cases: non-overlapping, overlapping; equivalent, and then a
  // couple permutations (switch left with right and make sure it still works).
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  Range z(0.0, 2.0);
  Range w;
  z &= y;
  w = x & y;

  BOOST_REQUIRE_SMALL(z.Width(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Width(), 1e-5);

  // Reverse operator precedence.
  z = y;
  z &= x;
  w = y & x;

  BOOST_REQUIRE_SMALL(z.Width(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Width(), 1e-5);

  // Now make them overlapping.
  x = Range(0.0, 3.5);
  y = Range(3.0, 4.0);

  z = x;
  z &= y;
  w = x & y;

  BOOST_REQUIRE_CLOSE(z.Lo(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 3.5, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 3.5, 1e-5);

  // Reverse operator precedence.
  z = y;
  z &= x;
  w = y & x;

  BOOST_REQUIRE_CLOSE(z.Lo(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 3.5, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 3.5, 1e-5);

  // Now make them equivalent.
  x = Range(2.0, 4.0);
  y = Range(2.0, 4.0);

  z = x;
  z &= y;
  w = x & y;

  BOOST_REQUIRE_CLOSE(z.Lo(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 4.0, 1e-5);
}

/**
 * Test multiplication of a range with a double.
 */
BOOST_AUTO_TEST_CASE(RangeMultiply)
{
  // We need to test both * and *=, as well as both cases of *.
  // We'll try with a couple of numbers: -1, 0, 2.
  // And we'll have a couple of cases for bounds: strictly less than zero;
  // including zero; and strictly greater than zero.
  //
  // So, nine total cases.
  Range x(-5.0, -3.0);
  Range y(-5.0, -3.0);
  Range z;
  Range w;

  y *= -1.0;
  z = x * -1.0;
  w = -1.0 * x;

  BOOST_REQUIRE_CLOSE(y.Lo(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(y.Hi(), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Lo(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 5.0, 1e-5);

  y = x;
  y *= 0.0;
  z = x * 0.0;
  w = 0.0 * x;

  BOOST_REQUIRE_SMALL(y.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(y.Hi(), 1e-5);
  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(z.Hi(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Hi(), 1e-5);

  y = x;
  y *= 2.0;
  z = x * 2.0;
  w = 2.0 * x;

  BOOST_REQUIRE_CLOSE(y.Lo(), -10.0, 1e-5);
  BOOST_REQUIRE_CLOSE(y.Hi(), -6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Lo(), -10.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), -6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), -10.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), -6.0, 1e-5);

  x = Range(-2.0, 2.0);
  y = x;

  y *= -1.0;
  z = x * -1.0;
  w = -1.0 * x;

  BOOST_REQUIRE_CLOSE(y.Lo(), -2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(y.Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Lo(), -2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), -2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 2.0, 1e-5);

  y = x;
  y *= 0.0;
  z = x * 0.0;
  w = 0.0 * x;

  BOOST_REQUIRE_SMALL(y.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(y.Hi(), 1e-5);
  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(z.Hi(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Hi(), 1e-5);

  y = x;
  y *= 2.0;
  z = x * 2.0;
  w = 2.0 * x;

  BOOST_REQUIRE_CLOSE(y.Lo(), -4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(y.Hi(), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Lo(), -4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), -4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 4.0, 1e-5);

  x = Range(3.0, 5.0);

  y = x;
  y *= -1.0;
  z = x * -1.0;
  w = -1.0 * x;

  BOOST_REQUIRE_CLOSE(y.Lo(), -5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(y.Hi(), -3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Lo(), -5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), -3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), -5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), -3.0, 1e-5);

  y = x;
  y *= 0.0;
  z = x * 0.0;
  w = 0.0 * x;

  BOOST_REQUIRE_SMALL(y.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(y.Hi(), 1e-5);
  BOOST_REQUIRE_SMALL(z.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(z.Hi(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Lo(), 1e-5);
  BOOST_REQUIRE_SMALL(w.Hi(), 1e-5);

  y = x;
  y *= 2.0;
  z = x * 2.0;
  w = 2.0 * x;

  BOOST_REQUIRE_CLOSE(y.Lo(), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(y.Hi(), 10.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Lo(), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(z.Hi(), 10.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Lo(), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(w.Hi(), 10.0, 1e-5);
}

/**
 * Test equality operator.
 */
BOOST_AUTO_TEST_CASE(RangeEquality)
{
  // Three cases: non-overlapping, overlapping, equivalent.  We should also
  // consider empty ranges, which are not necessarily equal...
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  // These are odd calls, but we don't want to use operator!= here.
  BOOST_REQUIRE_EQUAL((x == y), false);
  BOOST_REQUIRE_EQUAL((y == x), false);

  y = Range(1.0, 3.0);

  BOOST_REQUIRE_EQUAL((x == y), false);
  BOOST_REQUIRE_EQUAL((y == x), false);

  y = Range(0.0, 2.0);

  BOOST_REQUIRE_EQUAL((x == y), true);
  BOOST_REQUIRE_EQUAL((y == x), true);

  x = Range(1.0, -1.0); // Empty.
  y = Range(1.0, -1.0); // Also empty.

  BOOST_REQUIRE_EQUAL((x == y), true);
  BOOST_REQUIRE_EQUAL((y == x), true);

  // No need to test what it does if the empty ranges are different "ranges"
  // because we are not forcing behavior for that.
}

/**
 * Test inequality operator.
 */
BOOST_AUTO_TEST_CASE(RangeInequality)
{
  // We will use the same three cases as the RangeEquality test.
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  // Again, odd calls, but we want to force use of operator!=.
  BOOST_REQUIRE_EQUAL((x != y), true);
  BOOST_REQUIRE_EQUAL((y != x), true);

  y = Range(1.0, 3.0);

  BOOST_REQUIRE_EQUAL((x != y), true);
  BOOST_REQUIRE_EQUAL((y != x), true);

  y = Range(0.0, 2.0);

  BOOST_REQUIRE_EQUAL((x != y), false);
  BOOST_REQUIRE_EQUAL((y != x), false);

  x = Range(1.0, -1.0); // Empty.
  y = Range(1.0, -1.0); // Also empty.

  BOOST_REQUIRE_EQUAL((x != y), false);
  BOOST_REQUIRE_EQUAL((y != x), false);
}

/**
 * Test strict less-than operator.
 */
BOOST_AUTO_TEST_CASE(RangeStrictLessThan)
{
  // Three cases: non-overlapping, overlapping, and equivalent.
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  BOOST_REQUIRE_EQUAL((x < y), true);
  BOOST_REQUIRE_EQUAL((y < x), false);

  y = Range(1.0, 3.0);

  BOOST_REQUIRE_EQUAL((x < y), false);
  BOOST_REQUIRE_EQUAL((y < x), false);

  y = Range(0.0, 2.0);

  BOOST_REQUIRE_EQUAL((x < y), false);
  BOOST_REQUIRE_EQUAL((y < x), false);
}

/**
 * Test strict greater-than operator.
 */
BOOST_AUTO_TEST_CASE(RangeStrictGreaterThan)
{
  // Three cases: non-overlapping, overlapping, and equivalent.
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  BOOST_REQUIRE_EQUAL((x > y), false);
  BOOST_REQUIRE_EQUAL((y > x), true);

  y = Range(1.0, 3.0);

  BOOST_REQUIRE_EQUAL((x > y), false);
  BOOST_REQUIRE_EQUAL((y > x), false);

  y = Range(0.0, 2.0);

  BOOST_REQUIRE_EQUAL((x > y), false);
  BOOST_REQUIRE_EQUAL((y > x), false);
}

/**
 * Test the Contains() operator.
 */
BOOST_AUTO_TEST_CASE(RangeContains)
{
  // We have three Range cases: strictly less than 0; overlapping 0; and
  // strictly greater than 0.  Then the numbers we check can be the same three
  // cases, including one greater than and one less than the range.  This should
  // be about 15 total cases.
  Range x(-2.0, -1.0);

  BOOST_REQUIRE(!x.Contains(-3.0));
  BOOST_REQUIRE(x.Contains(-2.0));
  BOOST_REQUIRE(x.Contains(-1.5));
  BOOST_REQUIRE(x.Contains(-1.0));
  BOOST_REQUIRE(!x.Contains(-0.5));
  BOOST_REQUIRE(!x.Contains(0.0));
  BOOST_REQUIRE(!x.Contains(1.0));

  x = Range(-1.0, 1.0);

  BOOST_REQUIRE(!x.Contains(-2.0));
  BOOST_REQUIRE(x.Contains(-1.0));
  BOOST_REQUIRE(x.Contains(0.0));
  BOOST_REQUIRE(x.Contains(1.0));
  BOOST_REQUIRE(!x.Contains(2.0));

  x = Range(1.0, 2.0);

  BOOST_REQUIRE(!x.Contains(-1.0));
  BOOST_REQUIRE(!x.Contains(0.0));
  BOOST_REQUIRE(!x.Contains(0.5));
  BOOST_REQUIRE(x.Contains(1.0));
  BOOST_REQUIRE(x.Contains(1.5));
  BOOST_REQUIRE(x.Contains(2.0));
  BOOST_REQUIRE(!x.Contains(2.5));

  // Now let's try it on an empty range.
  x = Range();

  BOOST_REQUIRE(!x.Contains(-10.0));
  BOOST_REQUIRE(!x.Contains(0.0));
  BOOST_REQUIRE(!x.Contains(10.0));

  // And an infinite range.
  x = Range(-DBL_MAX, DBL_MAX);

  BOOST_REQUIRE(x.Contains(-10.0));
  BOOST_REQUIRE(x.Contains(0.0));
  BOOST_REQUIRE(x.Contains(10.0));
}

/**
 * Test that Range::Contains() works on other Ranges.  It should return false
 * unless the ranges overlap at all.
 */
BOOST_AUTO_TEST_CASE(RangeContainsRange)
{
  // Empty ranges should not contain each other.
  Range a;
  Range b;

  BOOST_REQUIRE_EQUAL(a.Contains(b), false);
  BOOST_REQUIRE_EQUAL(b.Contains(a), false);

  // Completely disparate ranges.
  a = Range(-5.0, -3.0);
  b = Range(3.0, 5.0);

  BOOST_REQUIRE_EQUAL(a.Contains(b), false);
  BOOST_REQUIRE_EQUAL(b.Contains(a), false);

  // Overlapping at the end-point; this is containment of the end point.
  a = Range(-5.0, 0.0);
  b = Range(0.0, 5.0);

  BOOST_REQUIRE_EQUAL(a.Contains(b), true);
  BOOST_REQUIRE_EQUAL(b.Contains(a), true);

  // Partially overlapping.
  a = Range(-5.0, 2.0);
  b = Range(-2.0, 5.0);

  BOOST_REQUIRE_EQUAL(a.Contains(b), true);
  BOOST_REQUIRE_EQUAL(b.Contains(a), true);

  // One range encloses the other.
  a = Range(-5.0, 5.0);
  b = Range(-3.0, 3.0);

  BOOST_REQUIRE_EQUAL(a.Contains(b), true);
  BOOST_REQUIRE_EQUAL(b.Contains(a), true);

  // Identical ranges.
  a = Range(-3.0, 3.0);
  b = Range(-3.0, 3.0);

  BOOST_REQUIRE_EQUAL(a.Contains(b), true);
  BOOST_REQUIRE_EQUAL(b.Contains(a), true);

  // Single-point ranges.
  a = Range(0.0, 0.0);
  b = Range(0.0, 0.0);

  BOOST_REQUIRE_EQUAL(a.Contains(b), true);
  BOOST_REQUIRE_EQUAL(b.Contains(a), true);
}

/**
 * Make sure shuffling data works.
 */
BOOST_AUTO_TEST_CASE(ShuffleTest)
{
  arma::mat data(3, 10, arma::fill::zeros);
  arma::Row<size_t> labels(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
  }

  arma::mat outputData;
  arma::Row<size_t> outputLabels;

  ShuffleData(data, labels, outputData, outputLabels);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_SMALL(outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL(outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);
}

/**
 * Make sure shuffling sparse data works.
 */
BOOST_AUTO_TEST_CASE(SparseShuffleTest)
{
  arma::sp_mat data(3, 10);
  arma::Row<size_t> labels(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
  }
  // This appears to be a necessary workaround for an Armadillo 8 bug.
  data *= 1.0;

  arma::sp_mat outputData;
  arma::Row<size_t> outputLabels;

  ShuffleData(data, labels, outputData, outputLabels);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_SMALL((double) outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL((double) outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);
}

/**
 * Make sure shuffling cubes works.
 */
BOOST_AUTO_TEST_CASE(CubeShuffleTest)
{
  arma::cube data(3, 10, 5, arma::fill::zeros);
  arma::cube labels(1, 10, 5);
  for (size_t i = 0; i < labels.n_slices; ++i)
  {
    for (size_t j = 0; j < labels.n_cols; ++j)
    {
      data(0, j, i) = i;
      data(1, j, i) = j;
      labels(0, j, i) = j + i;
    }
  }

  arma::cube outputData, outputLabels;

  ShuffleData(data, labels, outputData, outputLabels);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputData.n_slices, data.n_slices);
  BOOST_REQUIRE_EQUAL(outputLabels.n_rows, labels.n_rows);
  BOOST_REQUIRE_EQUAL(outputLabels.n_cols, labels.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_slices, labels.n_slices);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    for (size_t s = 0; s < data.n_slices; ++s)
    {
      BOOST_REQUIRE_EQUAL(data(0, i, s) + data(1, i, s), labels(0, i, s));
      BOOST_REQUIRE_SMALL(data(2, i, s), 1e-5);
      counts[data(1, i, s)]++;
    }
  }

  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], data.n_slices);
}

/**
 * Make sure shuffling data with weights works.
 */
BOOST_AUTO_TEST_CASE(ShuffleWeightsTest)
{
  arma::mat data(3, 10, arma::fill::zeros);
  arma::Row<size_t> labels(10);
  arma::rowvec weights(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
    weights[i] = i;
  }

  arma::mat outputData;
  arma::Row<size_t> outputLabels;
  arma::rowvec outputWeights;

  ShuffleData(data, labels, weights, outputData, outputLabels, outputWeights);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);
  BOOST_REQUIRE_EQUAL(outputWeights.n_elem, weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), (size_t) outputWeights[i]);
    BOOST_REQUIRE_SMALL(outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL(outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL(counts[i], 1);
    BOOST_REQUIRE_EQUAL(weightCounts[i], 1);
  }
}

/**
 * Make sure shuffling sparse data with weights works.
 */
BOOST_AUTO_TEST_CASE(SparseShuffleWeightsTest)
{
  arma::sp_mat data(3, 10);
  arma::Row<size_t> labels(10);
  arma::rowvec weights(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
    weights[i] = i;
  }
  // This appears to be a necessary workaround for an Armadillo 8 bug.
  data *= 1.0;

  arma::sp_mat outputData;
  arma::Row<size_t> outputLabels;
  arma::rowvec outputWeights;

  ShuffleData(data, labels, weights, outputData, outputLabels, outputWeights);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);
  BOOST_REQUIRE_EQUAL(outputWeights.n_elem, weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), (size_t) outputWeights[i]);
    BOOST_REQUIRE_SMALL((double) outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL((double) outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL(counts[i], 1);
    BOOST_REQUIRE_EQUAL(weightCounts[i], 1);
  }
}

/**
 * Make sure shuffling data works when the same matrices are given as input and
 * output.
 */
BOOST_AUTO_TEST_CASE(InplaceShuffleTest)
{
  arma::mat data(3, 10, arma::fill::zeros);
  arma::Row<size_t> labels(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
  }

  arma::mat outputData(data);
  arma::Row<size_t> outputLabels(labels);

  ShuffleData(outputData, outputLabels, outputData, outputLabels);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_SMALL(outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL(outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);
}

/**
 * Make sure shuffling sparse data works when the input and output matrices are
 * the same.
 */
BOOST_AUTO_TEST_CASE(InplaceSparseShuffleTest)
{
  arma::sp_mat data(3, 10);
  arma::Row<size_t> labels(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
  }

  arma::sp_mat outputData(data);
  arma::Row<size_t> outputLabels(labels);

  ShuffleData(outputData, outputLabels, outputData, outputLabels);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_SMALL((double) outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL((double) outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);
}

/**
 * Make sure shuffling cubes works when the input and output cubes are the same.
 */
BOOST_AUTO_TEST_CASE(InplaceCubeShuffleTest)
{
  arma::cube data(3, 10, 5, arma::fill::zeros);
  arma::cube labels(1, 10, 5);
  for (size_t i = 0; i < labels.n_slices; ++i)
  {
    for (size_t j = 0; j < labels.n_cols; ++j)
    {
      data(0, j, i) = i;
      data(1, j, i) = j;
      labels(0, j, i) = j + i;
    }
  }

  arma::cube outputData(data), outputLabels(labels);

  ShuffleData(outputData, outputLabels, outputData, outputLabels);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputData.n_slices, data.n_slices);
  BOOST_REQUIRE_EQUAL(outputLabels.n_rows, labels.n_rows);
  BOOST_REQUIRE_EQUAL(outputLabels.n_cols, labels.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_slices, labels.n_slices);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    for (size_t s = 0; s < data.n_slices; ++s)
    {
      BOOST_REQUIRE_EQUAL(data(0, i, s) + data(1, i, s), labels(0, i, s));
      BOOST_REQUIRE_SMALL(data(2, i, s), 1e-5);
      counts[data(1, i, s)]++;
    }
  }

  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], data.n_slices);
}

/**
 * Make sure shuffling data with weights works when the same matrices are given
 * as input and output.
 */
BOOST_AUTO_TEST_CASE(InplaceShuffleWeightsTest)
{
  arma::mat data(3, 10, arma::fill::zeros);
  arma::Row<size_t> labels(10);
  arma::rowvec weights(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
    weights[i] = i;
  }

  arma::mat outputData(data);
  arma::Row<size_t> outputLabels(labels);
  arma::rowvec outputWeights(weights);

  ShuffleData(outputData, outputLabels, outputWeights, outputData, outputLabels,
      outputWeights);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);
  BOOST_REQUIRE_EQUAL(outputWeights.n_elem, weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), (size_t) outputWeights[i]);
    BOOST_REQUIRE_SMALL(outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL(outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL(counts[i], 1);
    BOOST_REQUIRE_EQUAL(weightCounts[i], 1);
  }
}

/**
 * Make sure shuffling sparse data with weights works when the input and output
 * matrices are the same.
 */
BOOST_AUTO_TEST_CASE(InplaceSparseShuffleWeightsTest)
{
  arma::sp_mat data(3, 10);
  arma::Row<size_t> labels(10);
  arma::rowvec weights(10);
  for (size_t i = 0; i < 10; ++i)
  {
    data(0, i) = i;
    labels[i] = i;
    weights[i] = i;
  }

  arma::sp_mat outputData(data);
  arma::Row<size_t> outputLabels(labels);
  arma::rowvec outputWeights(weights);

  ShuffleData(outputData, outputLabels, outputWeights, outputData, outputLabels,
      outputWeights);

  BOOST_REQUIRE_EQUAL(outputData.n_rows, data.n_rows);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(outputLabels.n_elem, labels.n_elem);
  BOOST_REQUIRE_EQUAL(outputWeights.n_elem, weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), outputLabels[i]);
    BOOST_REQUIRE_EQUAL((size_t) outputData(0, i), (size_t) outputWeights[i]);
    BOOST_REQUIRE_SMALL((double) outputData(1, i), 1e-5);
    BOOST_REQUIRE_SMALL((double) outputData(2, i), 1e-5);
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL(counts[i], 1);
    BOOST_REQUIRE_EQUAL(weightCounts[i], 1);
  }
}

BOOST_AUTO_TEST_SUITE_END();
