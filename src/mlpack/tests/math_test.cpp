/**
 * @file tests/math_test.cpp
 * @author Ryan Curtin
 *
 * Tests for math functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;

/**
 * Verify that the empty constructor creates an empty range.
 */
TEST_CASE("RangeEmptyConstructor", "[MathTest]")
{
  Range x = Range();

  // Just verify that it is empty.
  REQUIRE(x.Lo() > x.Hi());
}

/**
 * Verify that the point constructor correctly creates a range that is just a
 * point.
 */
TEST_CASE("RangePointConstructor", "[MathTest]")
{
  Range x(10.0);

  REQUIRE(x.Lo() == Approx(x.Hi()).epsilon(1e-27));
  REQUIRE(x.Width() == Approx(0.0).margin(1e-5));
  REQUIRE(x.Lo() == Approx(10.0).epsilon(1e-27));
  REQUIRE(x.Hi() == Approx(10.0).epsilon(1e-27));
}

/**
 * Verify that the range constructor correctly creates the range.
 */
TEST_CASE("RangeConstructor", "[MathTest]")
{
  Range x(0.5, 5.5);

  REQUIRE(x.Lo() == Approx(0.5).epsilon(1e-27));
  REQUIRE(x.Hi() == Approx(5.5).epsilon(1e-27));
}

/**
 * Test that we get the width correct.
 */
TEST_CASE("RangeWidth", "[MathTest]")
{
  Range x(0.0, 10.0);

  REQUIRE(x.Width() == Approx(10.0).epsilon(1e-22));

  // Make it empty.
  x.Hi() = 0.0;

  REQUIRE(x.Width() == Approx(0.0).margin(1e-5));

  // Make it negative.
  x.Hi() = -2.0;

  REQUIRE(x.Width() == Approx(0.0).margin(1e-5));

  // Just one more test.
  x.Lo() = -5.2;
  x.Hi() = 5.2;

  REQUIRE(x.Width() == Approx(10.4).epsilon(1e-7));
}

/**
 * Test that we get the midpoint correct.
 */
TEST_CASE("RangeMidpoint", "[MathTest]")
{
  Range x(0.0, 10.0);

  REQUIRE(x.Mid() == Approx(5.0).epsilon(1e-7));

  x.Lo() = -5.0;

  REQUIRE(x.Mid() == Approx(2.5).epsilon(1e-7));
}

/**
 * Test that we can expand to include other ranges correctly.
 */
TEST_CASE("RangeIncludeOther", "[MathTest]")
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

  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(5.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(5.0).epsilon(1e-7));

  // Switch operator precedence.
  z = y;
  z |= x;
  w = y | x;

  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(5.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(5.0).epsilon(1e-7));

  // Now make them overlapping.
  x = Range(0.0, 3.5);
  y = Range(3.0, 4.0);

  z = x;
  z |= y;
  w = x | y;

  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(4.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(4.0).epsilon(1e-7));

  // Switch operator precedence.
  z = y;
  z |= x;
  w = y | x;

  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(4.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(4.0).epsilon(1e-7));

  // Now the equivalent case.
  x = Range(0.0, 2.0);
  y = Range(0.0, 2.0);

  z = x;
  z |= y;
  w = x | y;

  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(2.0).epsilon(1e-7));

  z = y;
  z |= x;
  w = y | x;

  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(2.0).epsilon(1e-7));
}

/**
 * Test that we can 'and' ranges correctly.
 */
TEST_CASE("RangeIntersectOther", "[MathTest]")
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

  REQUIRE(z.Width() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Width() == Approx(0.0).margin(1e-5));

  // Reverse operator precedence.
  z = y;
  z &= x;
  w = y & x;

  REQUIRE(z.Width() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Width() == Approx(0.0).margin(1e-5));

  // Now make them overlapping.
  x = Range(0.0, 3.5);
  y = Range(3.0, 4.0);

  z = x;
  z &= y;
  w = x & y;

  REQUIRE(z.Lo() == Approx(3.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(3.5).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(3.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(3.5).epsilon(1e-7));

  // Reverse operator precedence.
  z = y;
  z &= x;
  w = y & x;

  REQUIRE(z.Lo() == Approx(3.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(3.5).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(3.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(3.5).epsilon(1e-7));

  // Now make them equivalent.
  x = Range(2.0, 4.0);
  y = Range(2.0, 4.0);

  z = x;
  z &= y;
  w = x & y;

  REQUIRE(z.Lo() == Approx(2.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(4.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(2.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(4.0).epsilon(1e-7));
}

/**
 * Test multiplication of a range with a double.
 */
TEST_CASE("RangeMultiply", "[MathTest]")
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

  REQUIRE(y.Lo() == Approx(3.0).epsilon(1e-7));
  REQUIRE(y.Hi() == Approx(5.0).epsilon(1e-7));
  REQUIRE(z.Lo() == Approx(3.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(5.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(3.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(5.0).epsilon(1e-7));

  y = x;
  y *= 0.0;
  z = x * 0.0;
  w = 0.0 * x;

  REQUIRE(y.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(y.Hi() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(0.0).margin(1e-5));

  y = x;
  y *= 2.0;
  z = x * 2.0;
  w = 2.0 * x;

  REQUIRE(y.Lo() == Approx(-10.0).epsilon(1e-7));
  REQUIRE(y.Hi() == Approx(-6.0).epsilon(1e-7));
  REQUIRE(z.Lo() == Approx(-10.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(-6.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(-10.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(-6.0).epsilon(1e-7));

  x = Range(-2.0, 2.0);
  y = x;

  y *= -1.0;
  z = x * -1.0;
  w = -1.0 * x;

  REQUIRE(y.Lo() == Approx(-2.0).epsilon(1e-7));
  REQUIRE(y.Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(z.Lo() == Approx(-2.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(-2.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(2.0).epsilon(1e-7));

  y = x;
  y *= 0.0;
  z = x * 0.0;
  w = 0.0 * x;

  REQUIRE(y.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(y.Hi() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(0.0).margin(1e-5));

  y = x;
  y *= 2.0;
  z = x * 2.0;
  w = 2.0 * x;

  REQUIRE(y.Lo() == Approx(-4.0).epsilon(1e-7));
  REQUIRE(y.Hi() == Approx(4.0).epsilon(1e-7));
  REQUIRE(z.Lo() == Approx(-4.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(4.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(-4.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(4.0).epsilon(1e-7));

  x = Range(3.0, 5.0);

  y = x;
  y *= -1.0;
  z = x * -1.0;
  w = -1.0 * x;

  REQUIRE(y.Lo() == Approx(-5.0).epsilon(1e-7));
  REQUIRE(y.Hi() == Approx(-3.0).epsilon(1e-7));
  REQUIRE(z.Lo() == Approx(-5.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(-3.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(-5.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(-3.0).epsilon(1e-7));

  y = x;
  y *= 0.0;
  z = x * 0.0;
  w = 0.0 * x;

  REQUIRE(y.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(y.Hi() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(z.Hi() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(w.Hi() == Approx(0.0).margin(1e-5));

  y = x;
  y *= 2.0;
  z = x * 2.0;
  w = 2.0 * x;

  REQUIRE(y.Lo() == Approx(6.0).epsilon(1e-7));
  REQUIRE(y.Hi() == Approx(10.0).epsilon(1e-7));
  REQUIRE(z.Lo() == Approx(6.0).epsilon(1e-7));
  REQUIRE(z.Hi() == Approx(10.0).epsilon(1e-7));
  REQUIRE(w.Lo() == Approx(6.0).epsilon(1e-7));
  REQUIRE(w.Hi() == Approx(10.0).epsilon(1e-7));
}

/**
 * Test equality operator.
 */
TEST_CASE("RangeEquality", "[MathTest]")
{
  // Three cases: non-overlapping, overlapping, equivalent.  We should also
  // consider empty ranges, which are not necessarily equal...
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  // These are odd calls, but we don't want to use operator!= here.
  REQUIRE((x == y) == false);
  REQUIRE((y == x) == false);

  y = Range(1.0, 3.0);

  REQUIRE((x == y) == false);
  REQUIRE((y == x) == false);

  y = Range(0.0, 2.0);

  REQUIRE((x == y) == true);
  REQUIRE((y == x) == true);

  x = Range(1.0, -1.0); // Empty.
  y = Range(1.0, -1.0); // Also empty.

  REQUIRE((x == y) == true);
  REQUIRE((y == x) == true);

  // No need to test what it does if the empty ranges are different "ranges"
  // because we are not forcing behavior for that.
}

/**
 * Test inequality operator.
 */
TEST_CASE("RangeInequality", "[MathTest]")
{
  // We will use the same three cases as the RangeEquality test.
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  // Again, odd calls, but we want to force use of operator!=.
  REQUIRE((x != y) == true);
  REQUIRE((y != x) == true);

  y = Range(1.0, 3.0);

  REQUIRE((x != y) == true);
  REQUIRE((y != x) == true);

  y = Range(0.0, 2.0);

  REQUIRE((x != y) == false);
  REQUIRE((y != x) == false);

  x = Range(1.0, -1.0); // Empty.
  y = Range(1.0, -1.0); // Also empty.

  REQUIRE((x != y) == false);
  REQUIRE((y != x) == false);
}

/**
 * Test strict less-than operator.
 */
TEST_CASE("RangeStrictLessThan", "[MathTest]")
{
  // Three cases: non-overlapping, overlapping, and equivalent.
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  REQUIRE((x < y) == true);
  REQUIRE((y < x) == false);

  y = Range(1.0, 3.0);

  REQUIRE((x < y) == false);
  REQUIRE((y < x) == false);

  y = Range(0.0, 2.0);

  REQUIRE((x < y) == false);
  REQUIRE((y < x) == false);
}

/**
 * Test strict greater-than operator.
 */
TEST_CASE("RangeStrictGreaterThan", "[MathTest]")
{
  // Three cases: non-overlapping, overlapping, and equivalent.
  Range x(0.0, 2.0);
  Range y(3.0, 5.0);

  REQUIRE((x > y) == false);
  REQUIRE((y > x) == true);

  y = Range(1.0, 3.0);

  REQUIRE((x > y) == false);
  REQUIRE((y > x) == false);

  y = Range(0.0, 2.0);

  REQUIRE((x > y) == false);
  REQUIRE((y > x) == false);
}

/**
 * Test the Contains() operator.
 */
TEST_CASE("RangeContains", "[MathTest]")
{
  // We have three Range cases: strictly less than 0; overlapping 0; and
  // strictly greater than 0.  Then the numbers we check can be the same three
  // cases, including one greater than and one less than the range.  This should
  // be about 15 total cases.
  Range x(-2.0, -1.0);

  REQUIRE(!x.Contains(-3.0));
  REQUIRE(x.Contains(-2.0));
  REQUIRE(x.Contains(-1.5));
  REQUIRE(x.Contains(-1.0));
  REQUIRE(!x.Contains(-0.5));
  REQUIRE(!x.Contains(0.0));
  REQUIRE(!x.Contains(1.0));

  x = Range(-1.0, 1.0);

  REQUIRE(!x.Contains(-2.0));
  REQUIRE(x.Contains(-1.0));
  REQUIRE(x.Contains(0.0));
  REQUIRE(x.Contains(1.0));
  REQUIRE(!x.Contains(2.0));

  x = Range(1.0, 2.0);

  REQUIRE(!x.Contains(-1.0));
  REQUIRE(!x.Contains(0.0));
  REQUIRE(!x.Contains(0.5));
  REQUIRE(x.Contains(1.0));
  REQUIRE(x.Contains(1.5));
  REQUIRE(x.Contains(2.0));
  REQUIRE(!x.Contains(2.5));

  // Now let's try it on an empty range.
  x = Range();

  REQUIRE(!x.Contains(-10.0));
  REQUIRE(!x.Contains(0.0));
  REQUIRE(!x.Contains(10.0));

  // And an infinite range.
  x = Range(-DBL_MAX, DBL_MAX);

  REQUIRE(x.Contains(-10.0));
  REQUIRE(x.Contains(0.0));
  REQUIRE(x.Contains(10.0));
}

/**
 * Test that Range::Contains() works on other Ranges.  It should return false
 * unless the ranges overlap at all.
 */
TEST_CASE("RangeContainsRange", "[MathTest]")
{
  // Empty ranges should not contain each other.
  Range a;
  Range b;

  REQUIRE(a.Contains(b) == false);
  REQUIRE(b.Contains(a) == false);

  // Completely disparate ranges.
  a = Range(-5.0, -3.0);
  b = Range(3.0, 5.0);

  REQUIRE(a.Contains(b) == false);
  REQUIRE(b.Contains(a) == false);

  // Overlapping at the end-point; this is containment of the end point.
  a = Range(-5.0, 0.0);
  b = Range(0.0, 5.0);

  REQUIRE(a.Contains(b) == true);
  REQUIRE(b.Contains(a) == true);

  // Partially overlapping.
  a = Range(-5.0, 2.0);
  b = Range(-2.0, 5.0);

  REQUIRE(a.Contains(b) == true);
  REQUIRE(b.Contains(a) == true);

  // One range encloses the other.
  a = Range(-5.0, 5.0);
  b = Range(-3.0, 3.0);

  REQUIRE(a.Contains(b) == true);
  REQUIRE(b.Contains(a) == true);

  // Identical ranges.
  a = Range(-3.0, 3.0);
  b = Range(-3.0, 3.0);

  REQUIRE(a.Contains(b) == true);
  REQUIRE(b.Contains(a) == true);

  // Single-point ranges.
  a = Range(0.0, 0.0);
  b = Range(0.0, 0.0);

  REQUIRE(a.Contains(b) == true);
  REQUIRE(b.Contains(a) == true);
}

/**
 * Make sure shuffling data works.
 */
TEST_CASE("ShuffleTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE(outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE(outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    REQUIRE(counts[i] == 1);
}

/**
 * Make sure shuffling sparse data works.
 */
TEST_CASE("SparseShuffleTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE((double) outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE((double) outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    REQUIRE(counts[i] == 1);
}

/**
 * Make sure shuffling cubes works.
 */
TEST_CASE("CubeShuffleTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputData.n_slices == data.n_slices);
  REQUIRE(outputLabels.n_rows == labels.n_rows);
  REQUIRE(outputLabels.n_cols == labels.n_cols);
  REQUIRE(outputLabels.n_slices == labels.n_slices);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    for (size_t s = 0; s < data.n_slices; ++s)
    {
      REQUIRE(data(0, i, s) + data(1, i, s) == labels(0, i, s));
      REQUIRE(data(2, i, s) == Approx(0.0).margin(1e-5));
      counts[data(1, i, s)]++;
    }
  }

  for (size_t i = 0; i < 10; ++i)
    REQUIRE(counts[i] == data.n_slices);
}

/**
 * Make sure shuffling data with weights works.
 */
TEST_CASE("ShuffleWeightsTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);
  REQUIRE(outputWeights.n_elem == weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE((size_t) outputData(0, i) == (size_t) outputWeights[i]);
    REQUIRE(outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE(outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(counts[i] == 1);
    REQUIRE(weightCounts[i] == 1);
  }
}

/**
 * Make sure shuffling sparse data with weights works.
 */
TEST_CASE("SparseShuffleWeightsTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);
  REQUIRE(outputWeights.n_elem == weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE((size_t) outputData(0, i) == (size_t) outputWeights[i]);
    REQUIRE((double) outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE((double) outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(counts[i] == 1);
    REQUIRE(weightCounts[i] == 1);
  }
}

/**
 * Make sure shuffling data works when the same matrices are given as input and
 * output.
 */
TEST_CASE("InplaceShuffleTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE(outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE(outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    REQUIRE(counts[i] == 1);
}

/**
 * Make sure shuffling sparse data works when the input and output matrices are
 * the same.
 */
TEST_CASE("InplaceSparseShuffleTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE((double) outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE((double) outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
    REQUIRE(counts[i] == 1);
}

/**
 * Make sure shuffling cubes works when the input and output cubes are the same.
 */
TEST_CASE("InplaceCubeShuffleTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputData.n_slices == data.n_slices);
  REQUIRE(outputLabels.n_rows == labels.n_rows);
  REQUIRE(outputLabels.n_cols == labels.n_cols);
  REQUIRE(outputLabels.n_slices == labels.n_slices);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    for (size_t s = 0; s < data.n_slices; ++s)
    {
      REQUIRE(data(0, i, s) + data(1, i, s) == labels(0, i, s));
      REQUIRE(data(2, i, s) == Approx(0.0).margin(1e-5));
      counts[data(1, i, s)]++;
    }
  }

  for (size_t i = 0; i < 10; ++i)
    REQUIRE(counts[i] == data.n_slices);
}

/**
 * Make sure shuffling data with weights works when the same matrices are given
 * as input and output.
 */
TEST_CASE("InplaceShuffleWeightsTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);
  REQUIRE(outputWeights.n_elem == weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE((size_t) outputData(0, i) == (size_t) outputWeights[i]);
    REQUIRE(outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE(outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(counts[i] == 1);
    REQUIRE(weightCounts[i] == 1);
  }
}

/**
 * Make sure shuffling sparse data with weights works when the input and output
 * matrices are the same.
 */
TEST_CASE("InplaceSparseShuffleWeightsTest", "[MathTest]")
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

  REQUIRE(outputData.n_rows == data.n_rows);
  REQUIRE(outputData.n_cols == data.n_cols);
  REQUIRE(outputLabels.n_elem == labels.n_elem);
  REQUIRE(outputWeights.n_elem == weights.n_elem);

  // Make sure we only have each point once.
  arma::Row<size_t> counts(10, arma::fill::zeros);
  arma::Row<size_t> weightCounts(10, arma::fill::zeros);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE((size_t) outputData(0, i) == outputLabels[i]);
    REQUIRE((size_t) outputData(0, i) == (size_t) outputWeights[i]);
    REQUIRE((double) outputData(1, i) == Approx(0.0).margin(1e-5));
    REQUIRE((double) outputData(2, i) == Approx(0.0).margin(1e-5));
    counts[outputLabels[i]]++;
    weightCounts[(size_t) outputWeights[i]]++;
  }

  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(counts[i] == 1);
    REQUIRE(weightCounts[i] == 1);
  }
}
