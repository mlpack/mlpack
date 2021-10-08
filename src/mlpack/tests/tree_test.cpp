/**
 * @file tests/tree_test.cpp
 *
 * Tests for tree-building methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>
#include <mlpack/core/tree/cover_tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>

#include <queue>
#include <stack>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

/**
 * Ensure that a bound, by default, is empty and has no dimensionality.
 */
TEST_CASE("HRectBoundEmptyConstructor", "[TreeTest]")
{
  HRectBound<EuclideanDistance> b;

  REQUIRE((int) b.Dim() == 0);
  REQUIRE(b.MinWidth() == 0.0);
}

/**
 * Ensure that when we specify the dimensionality in the constructor, it is
 * correct, and the bounds are all the empty set.
 */
TEST_CASE("HRectBoundDimConstructor", "[TreeTest]")
{
  HRectBound<EuclideanDistance> b(2); // We'll do this with 2 and 5 dimensions.

  REQUIRE(b.Dim() == 2);
  REQUIRE(b[0].Width() == Approx(0.0).margin(1e-5));
  REQUIRE(b[1].Width() == Approx(0.0).margin(1e-5));

  b = HRectBound<EuclideanDistance>(5);

  REQUIRE(b.Dim() == 5);
  REQUIRE(b[0].Width() == Approx(0.0).margin(1e-5));
  REQUIRE(b[1].Width() == Approx(0.0).margin(1e-5));
  REQUIRE(b[2].Width() == Approx(0.0).margin(1e-5));
  REQUIRE(b[3].Width() == Approx(0.0).margin(1e-5));
  REQUIRE(b[4].Width() == Approx(0.0).margin(1e-5));

  REQUIRE(b.MinWidth() == 0.0);
}

/**
 * Test the copy constructor.
 */
TEST_CASE("HRectBoundCopyConstructor", "[TreeTest]")
{
  HRectBound<EuclideanDistance> b(2);
  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 3.0);
  b.MinWidth() = 0.5;

  HRectBound<EuclideanDistance> c(b);

  REQUIRE(c.Dim() == 2);
  REQUIRE(c[0].Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(c[0].Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(c[1].Lo() == Approx(2.0).epsilon(1e-7));
  REQUIRE(c[1].Hi() == Approx(3.0).epsilon(1e-7));
  REQUIRE(c.MinWidth() == Approx(0.5).epsilon(1e-7));
}

/**
 * Test the assignment operator.
 */
TEST_CASE("HRectBoundAssignmentOperator", "[TreeTest]")
{
  HRectBound<EuclideanDistance> b(2);
  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 3.0);
  b.MinWidth() = 0.5;

  HRectBound<EuclideanDistance> c(4);

  c = b;

  REQUIRE(c.Dim() == 2);
  REQUIRE(c[0].Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(c[0].Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(c[1].Lo() == Approx(2.0).epsilon(1e-7));
  REQUIRE(c[1].Hi() == Approx(3.0).epsilon(1e-7));
  REQUIRE(c.MinWidth() == Approx(0.5).epsilon(1e-7));
}

/**
 * Test that clearing the dimensions resets the bound to empty.
 */
TEST_CASE("HRectBoundClear", "[TreeTest]")
{
  HRectBound<EuclideanDistance> b(2); // We'll do this with two dimensions only.

  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 4.0);
  b.MinWidth() = 1.0;

  // Now we just need to make sure that we clear the range.
  b.Clear();

  REQUIRE(b[0].Width() == Approx(0.0).margin(1e-5));
  REQUIRE(b[1].Width() == Approx(0.0).margin(1e-5));
  REQUIRE(b.MinWidth() == Approx(0.0).margin(1e-5));
}

TEST_CASE("HRectBoundMoveConstructor", "[TreeTest]")
{
  HRectBound<EuclideanDistance> b(2);
  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 4.0);
  b.MinWidth() = 1.0;

  HRectBound<EuclideanDistance> b2(std::move(b));

  REQUIRE(b.Dim() == 0);
  REQUIRE(b2.Dim() == 2);

  REQUIRE(b.MinWidth() == 0.0);
  REQUIRE(b2.MinWidth() == 1.0);

  REQUIRE(b2[0].Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(b2[0].Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(b2[1].Lo() == Approx(2.0).epsilon(1e-7));
  REQUIRE(b2[1].Hi() == Approx(4.0).epsilon(1e-7));
}

/**
 * Ensure that we get the correct center for our bound.
 */
TEST_CASE("HRectBoundCenter", "[TreeTest]")
{
  // Create a simple 3-dimensional bound.
  HRectBound<EuclideanDistance> b(3);

  b[0] = Range(0.0, 5.0);
  b[1] = Range(-2.0, -1.0);
  b[2] = Range(-10.0, 50.0);

  arma::vec center;

  b.Center(center);

  REQUIRE(center.n_elem == 3);
  REQUIRE(center[0] == Approx(2.5).epsilon(1e-7));
  REQUIRE(center[1] == Approx(-1.5).epsilon(1e-7));
  REQUIRE(center[2] == Approx(20.0).epsilon(1e-7));
}

/**
 * Ensure the volume calculation is correct.
 */
TEST_CASE("HRectBoundVolume", "[TreeTest]")
{
  // Create a simple 3-dimensional bound.
  HRectBound<EuclideanDistance> b(3);

  b[0] = Range(0.0, 5.0);
  b[1] = Range(-2.0, -1.0);
  b[2] = Range(-10.0, 50.0);

  REQUIRE(b.Volume() == Approx(300.0).epsilon(1e-7));
}

/**
 * Ensure that we calculate the correct minimum distance between a point and a
 * bound.
 */
TEST_CASE("HRectBoundMinDistancePoint", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  REQUIRE(b.MinDistance(point) == Approx(sqrt(95.0)).epsilon(1e-7));

  point = "2.0 5.0 2.0 -5.0 1.0";

  REQUIRE(b.MinDistance(point) == Approx(0.0).margin(1e-5));

  point = "1.0 2.0 0.0 -2.0 1.5";

  REQUIRE(b.MinDistance(point) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure that we calculate the correct minimum distance between a bound and
 * another bound.
 */
TEST_CASE("HRectBoundMinDistanceBound", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<EuclideanDistance> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  REQUIRE(b.MinDistance(c) == Approx(sqrt(22.0)).epsilon(1e-7));
  REQUIRE(c.MinDistance(b) == Approx(sqrt(22.0)).epsilon(1e-7));

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  REQUIRE(b.MinDistance(c) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(b) == Approx(0.0).margin(1e-5));

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  REQUIRE(b.MinDistance(c) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(b) == Approx(0.0).margin(1e-5));

  // The other bound fully overlaps the bound.
  REQUIRE(b.MinDistance(b) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(c) == Approx(0.0).margin(1e-5));

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  REQUIRE(b.MinDistance(c) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(b) == Approx(0.0).margin(1e-5));

  // Now we must be sure that the minimum distance to itself is 0.
  REQUIRE(b.MinDistance(b) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(c) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and a
 * point.  This uses the same test cases as the MinDistance test.
 */
TEST_CASE("HRectBoundMaxDistancePoint", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  REQUIRE(b.MaxDistance(point) == Approx(sqrt(253.0)).epsilon(1e-7));

  point = "2.0 5.0 2.0 -5.0 1.0";

  REQUIRE(b.MaxDistance(point) == Approx(sqrt(46.0)).epsilon(1e-7));

  point = "1.0 2.0 0.0 -2.0 1.5";

  REQUIRE(b.MaxDistance(point) == Approx(sqrt(23.25)).epsilon(1e-7));
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and
 * another bound.  This uses the same test cases as the MinDistance test.
 */
TEST_CASE("HRectBoundMaxDistanceBound", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<EuclideanDistance> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(210.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(210.0)).epsilon(1e-7));

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(134.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(134.0)).epsilon(1e-7));

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(102.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(102.0)).epsilon(1e-7));

  // The other bound fully overlaps the bound.
  REQUIRE(b.MaxDistance(b) == Approx(sqrt(46.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(c) == Approx(sqrt(61.0)).epsilon(1e-7));

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(100.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(100.0)).epsilon(1e-7));

  // Identical bounds.  This will be the sum of the squared widths in each
  // dimension.
  REQUIRE(b.MaxDistance(b) == Approx(sqrt(46.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(c) == Approx(sqrt(162.0)).epsilon(1e-7));

  // One last additional case.  If the bound encloses only one point, the
  // maximum distance between it and itself is 0.
  HRectBound<EuclideanDistance> d(2);

  d[0] = Range(2.0, 2.0);
  d[1] = Range(3.0, 3.0);

  REQUIRE(d.MaxDistance(d) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the behavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.
 */
TEST_CASE("HRectBoundRangeDistanceBound", "[TreeTest]")
{
  for (int i = 0; i < 50; ++i)
  {
    size_t dim = math::RandInt(20);

    HRectBound<EuclideanDistance> a(dim);
    HRectBound<EuclideanDistance> b(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec loA(dim);
    arma::vec widthA(dim);

    loA.randu();
    widthA.randu();

    arma::vec lo_b(dim);
    arma::vec width_b(dim);

    lo_b.randu();
    width_b.randu();

    for (size_t j = 0; j < dim; ++j)
    {
      a[j] = Range(loA[j], loA[j] + widthA[j]);
      b[j] = Range(lo_b[j], lo_b[j] + width_b[j]);
    }

    // Now ensure that MinDistance and MaxDistance report the same.
    Range r = a.RangeDistance(b);
    Range s = b.RangeDistance(a);

    REQUIRE(r.Lo() == Approx(s.Lo()).epsilon(1e-7));
    REQUIRE(r.Hi() == Approx(s.Hi()).epsilon(1e-7));

    REQUIRE(r.Lo() == Approx(a.MinDistance(b)).epsilon(1e-7));
    REQUIRE(r.Hi() == Approx(a.MaxDistance(b)).epsilon(1e-7));

    REQUIRE(s.Lo() == Approx(b.MinDistance(a)).epsilon(1e-7));
    REQUIRE(s.Hi() == Approx(b.MaxDistance(a)).epsilon(1e-7));
  }
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the bheavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.  This is for the
 * bound-to-point case.
 */
TEST_CASE("HRectBoundRangeDistancePoint", "[TreeTest]")
{
  for (int i = 0; i < 20; ++i)
  {
    size_t dim = math::RandInt(20);

    HRectBound<EuclideanDistance> a(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec loA(dim);
    arma::vec widthA(dim);

    loA.randu();
    widthA.randu();

    for (size_t j = 0; j < dim; ++j)
      a[j] = Range(loA[j], loA[j] + widthA[j]);

    // Now run the test on a few points.
    for (int j = 0; j < 10; ++j)
    {
      arma::vec point(dim);

      point.randu();

      Range r = a.RangeDistance(point);

      REQUIRE(r.Lo() == Approx(a.MinDistance(point)).epsilon(1e-7));
      REQUIRE(r.Hi() == Approx(a.MaxDistance(point)).epsilon(1e-7));
    }
  }
}

/**
 * Test that we can expand the bound to include a new point.
 */
TEST_CASE("HRectBoundOrOperatorPoint", "[TreeTest]")
{
  // Because this should be independent in each dimension, we can essentially
  // run five test cases at once.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(1.0, 3.0);
  b[1] = Range(2.0, 4.0);
  b[2] = Range(-2.0, -1.0);
  b[3] = Range(0.0, 0.0);
  b[4] = Range(); // Empty range.
  b.MinWidth() = 0.0;

  arma::vec point = "2.0 4.0 2.0 -1.0 6.0";

  b |= point;

  REQUIRE(b[0].Lo() == Approx(1.0).epsilon(1e-7));
  REQUIRE(b[0].Hi() == Approx(3.0).epsilon(1e-7));
  REQUIRE(b[1].Lo() == Approx(2.0).epsilon(1e-7));
  REQUIRE(b[1].Hi() == Approx(4.0).epsilon(1e-7));
  REQUIRE(b[2].Lo() == Approx(-2.0).epsilon(1e-7));
  REQUIRE(b[2].Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(b[3].Lo() == Approx(-1.0).epsilon(1e-7));
  REQUIRE(b[3].Hi() == Approx(0.0).margin(1e-5));
  REQUIRE(b[4].Lo() == Approx(6.0).epsilon(1e-7));
  REQUIRE(b[4].Hi() == Approx(6.0).epsilon(1e-7));
  REQUIRE(b.MinWidth() == Approx(0.0).margin(1e-5));
}

/**
 * Test that we can expand the bound to include another bound.
 */
TEST_CASE("HRectBoundOrOperatorBound", "[TreeTest]")
{
  // Because this should be independent in each dimension, we can run many tests
  // at once.
  HRectBound<EuclideanDistance> b(8);

  b[0] = Range(1.0, 3.0);
  b[1] = Range(2.0, 4.0);
  b[2] = Range(-2.0, -1.0);
  b[3] = Range(4.0, 5.0);
  b[4] = Range(2.0, 4.0);
  b[5] = Range(0.0, 0.0);
  b[6] = Range();
  b[7] = Range(1.0, 3.0);

  HRectBound<EuclideanDistance> c(8);

  c[0] = Range(-3.0, -1.0); // Entirely less than the other bound.
  c[1] = Range(0.0, 2.0); // Touching edges.
  c[2] = Range(-3.0, -1.5); // Partially overlapping.
  c[3] = Range(4.0, 5.0); // Identical.
  c[4] = Range(1.0, 5.0); // Entirely enclosing.
  c[5] = Range(2.0, 2.0); // A single point.
  c[6] = Range(1.0, 3.0);
  c[7] = Range(); // Empty set.

  HRectBound<EuclideanDistance> d = c;

  b |= c;
  d |= b;

  REQUIRE(b[0].Lo() == Approx(-3.0).epsilon(1e-7));
  REQUIRE(b[0].Hi() == Approx(3.0).epsilon(1e-7));
  REQUIRE(d[0].Lo() == Approx(-3.0).epsilon(1e-7));
  REQUIRE(d[0].Hi() == Approx(3.0).epsilon(1e-7));

  REQUIRE(b[1].Lo() == Approx(0.0).epsilon(1e-7));
  REQUIRE(b[1].Hi() == Approx(4.0).epsilon(1e-7));
  REQUIRE(d[1].Lo() == Approx(0.0).epsilon(1e-7));
  REQUIRE(d[1].Hi() == Approx(4.0).epsilon(1e-7));

  REQUIRE(b[2].Lo() == Approx(-3.0).epsilon(1e-7));
  REQUIRE(b[2].Hi() == Approx(-1.0).epsilon(1e-7));
  REQUIRE(d[2].Lo() == Approx(-3.0).epsilon(1e-7));
  REQUIRE(d[2].Hi() == Approx(-1.0).epsilon(1e-7));

  REQUIRE(b[3].Lo() == Approx(4.0).epsilon(1e-7));
  REQUIRE(b[3].Hi() == Approx(5.0).epsilon(1e-7));
  REQUIRE(d[3].Lo() == Approx(4.0).epsilon(1e-7));
  REQUIRE(d[3].Hi() == Approx(5.0).epsilon(1e-7));

  REQUIRE(b[4].Lo() == Approx(1.0).epsilon(1e-7));
  REQUIRE(b[4].Hi() == Approx(5.0).epsilon(1e-7));
  REQUIRE(d[4].Lo() == Approx(1.0).epsilon(1e-7));
  REQUIRE(d[4].Hi() == Approx(5.0).epsilon(1e-7));

  REQUIRE(b[5].Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(b[5].Hi() == Approx(2.0).epsilon(1e-7));
  REQUIRE(d[5].Lo() == Approx(0.0).margin(1e-5));
  REQUIRE(d[5].Hi() == Approx(2.0).epsilon(1e-7));

  REQUIRE(b[6].Lo() == Approx(1.0).epsilon(1e-7));
  REQUIRE(b[6].Hi() == Approx(3.0).epsilon(1e-7));
  REQUIRE(d[6].Lo() == Approx(1.0).epsilon(1e-7));
  REQUIRE(d[6].Hi() == Approx(3.0).epsilon(1e-7));

  REQUIRE(b[7].Lo() == Approx(1.0).epsilon(1e-7));
  REQUIRE(b[7].Hi() == Approx(3.0).epsilon(1e-7));
  REQUIRE(d[7].Lo() == Approx(1.0).epsilon(1e-7));
  REQUIRE(d[7].Hi() == Approx(3.0).epsilon(1e-7));

  REQUIRE(b.MinWidth() == Approx(1.0).epsilon(1e-7));
  REQUIRE(d.MinWidth() == Approx(1.0).epsilon(1e-7));
}

/**
 * Test that the Contains() function correctly figures out whether or not a
 * point is in a bound.
 */
TEST_CASE("HRectBoundContains", "[TreeTest]")
{
  // We can test a couple different points: completely outside the bound,
  // adjacent in one dimension to the bound, adjacent in all dimensions to the
  // bound, and inside the bound.
  HRectBound<EuclideanDistance> b(3);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(0.0, 2.0);
  b[2] = Range(0.0, 2.0);

  // Completely outside the range.
  arma::vec point = "-1.0 4.0 4.0";
  REQUIRE(!b.Contains(point));

  // Completely outside, but one dimension is in the range.
  point = "-1.0 4.0 1.0";
  REQUIRE(!b.Contains(point));

  // Outside, but one dimension is on the edge.
  point = "-1.0 0.0 3.0";
  REQUIRE(!b.Contains(point));

  // Two dimensions are on the edge, but one is outside.
  point = "0.0 0.0 3.0";
  REQUIRE(!b.Contains(point));

  // Completely on the edge (should be contained).
  point = "0.0 0.0 0.0";
  REQUIRE(b.Contains(point));

  // Inside the range.
  point = "0.3 1.0 0.4";
  REQUIRE(b.Contains(point));
}

TEST_CASE("TestBallBound", "[TreeTest]")
{
  BallBound<> b1;
  BallBound<> b2;

  // Create two balls with a center distance of 1 from each other.
  // Give the first one a radius of 0.3 and the second a radius of 0.4.
  b1.Center().set_size(3);
  b1.Center()[0] = 1;
  b1.Center()[1] = 2;
  b1.Center()[2] = 3;
  b1.Radius() = 0.3;

  b2.Center().set_size(3);
  b2.Center()[0] = 1;
  b2.Center()[1] = 2;
  b2.Center()[2] = 4;
  b2.Radius() = 0.4;

  REQUIRE(b1.MinDistance(b2) == Approx(1-0.3-0.4).epsilon(1e-7));
  REQUIRE(b1.RangeDistance(b2).Hi() == Approx(1+0.3+0.4).epsilon(1e-7));
  REQUIRE(b1.RangeDistance(b2).Lo() == Approx(1-0.3-0.4).epsilon(1e-7));
  REQUIRE(b1.RangeDistance(b2).Hi() == Approx(1+0.3+0.4).epsilon(1e-7));
  REQUIRE(b1.RangeDistance(b2).Lo() == Approx(1-0.3-0.4).epsilon(1e-7));

  REQUIRE(b2.MinDistance(b1) == Approx(1-0.3-0.4).epsilon(1e-7));
  REQUIRE(b2.MaxDistance(b1) == Approx(1+0.3+0.4).epsilon(1e-7));
  REQUIRE(b2.RangeDistance(b1).Hi() == Approx(1+0.3+0.4).epsilon(1e-7));
  REQUIRE(b2.RangeDistance(b1).Lo() == Approx(1-0.3-0.4).epsilon(1e-7));

  REQUIRE(b1.Contains(b1.Center()));
  REQUIRE(!b1.Contains(b2.Center()));

  REQUIRE(!b2.Contains(b1.Center()));
  REQUIRE(b2.Contains(b2.Center()));
  arma::vec b2point(3); // A point that's within the radius but not the center.
  b2point[0] = 1.1;
  b2point[1] = 2.1;
  b2point[2] = 4.1;

  REQUIRE(b2.Contains(b2point));

  REQUIRE(b1.MinDistance(b1.Center()) == Approx(0.0).margin(1e-5));
  REQUIRE(b1.MinDistance(b2.Center()) == Approx(1 - 0.3).epsilon(1e-7));
  REQUIRE(b2.MinDistance(b1.Center()) == Approx(1 - 0.4).epsilon(1e-7));
  REQUIRE(b2.MaxDistance(b1.Center()) == Approx(1 + 0.4).epsilon(1e-7));
  REQUIRE(b1.MaxDistance(b2.Center()) == Approx(1 + 0.3).epsilon(1e-7));
}

TEST_CASE("BallBoundMoveConstructor", "[TreeTest]")
{
  BallBound<> b1(2.0, arma::vec("2 1 1"));
  BallBound<> b2(std::move(b1));

  REQUIRE(b2.Dim() == 3);
  REQUIRE(b1.Dim() == 0);

  REQUIRE(b2.Center()[0] == Approx(2.0).epsilon(1e-7));
  REQUIRE(b2.Center()[1] == Approx(1.0).epsilon(1e-7));
  REQUIRE(b2.Center()[2] == Approx(1.0).epsilon(1e-7));

  REQUIRE(b2.MinWidth() == Approx(4.0).epsilon(1e-7));
  REQUIRE(b1.MinWidth() == Approx(0.0).margin(1e-5));
}

/**
 * Ensure that we calculate the correct minimum distance between a point and a
 * bound.
 */
TEST_CASE("HRectBoundRootMinDistancePoint", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  REQUIRE(b.MinDistance(point) == Approx(sqrt(95.0)).epsilon(1e-7));

  point = "2.0 5.0 2.0 -5.0 1.0";

  REQUIRE(b.MinDistance(point) == Approx(0.0).margin(1e-5));

  point = "1.0 2.0 0.0 -2.0 1.5";

  REQUIRE(b.MinDistance(point) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure that we calculate the correct minimum distance between a bound and
 * another bound.
 */
TEST_CASE("HRectBoundRootMinDistanceBound", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<EuclideanDistance> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  REQUIRE(b.MinDistance(c) == Approx(sqrt(22.0)).epsilon(1e-7));
  REQUIRE(c.MinDistance(b) == Approx(sqrt(22.0)).epsilon(1e-7));

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  REQUIRE(b.MinDistance(c) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(b) == Approx(0.0).margin(1e-5));

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  REQUIRE(b.MinDistance(c) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(b) == Approx(0.0).margin(1e-5));

  // The other bound fully overlaps the bound.
  REQUIRE(b.MinDistance(b) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(c) == Approx(0.0).margin(1e-5));

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  REQUIRE(b.MinDistance(c) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(b) == Approx(0.0).margin(1e-5));

  // Now we must be sure that the minimum distance to itself is 0.
  REQUIRE(b.MinDistance(b) == Approx(0.0).margin(1e-5));
  REQUIRE(c.MinDistance(c) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and a
 * point.  This uses the same test cases as the MinDistance test.
 */
TEST_CASE("HRectBoundRootMaxDistancePoint", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  REQUIRE(b.MaxDistance(point) == Approx(sqrt(253.0)).epsilon(1e-7));

  point = "2.0 5.0 2.0 -5.0 1.0";

  REQUIRE(b.MaxDistance(point) == Approx(sqrt(46.0)).epsilon(1e-7));

  point = "1.0 2.0 0.0 -2.0 1.5";

  REQUIRE(b.MaxDistance(point) == Approx(sqrt(23.25)).epsilon(1e-7));
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and
 * another bound.  This uses the same test cases as the MinDistance test.
 */
TEST_CASE("HRectBoundRootMaxDistanceBound", "[TreeTest]")
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<EuclideanDistance> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<EuclideanDistance> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(210.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(210.0)).epsilon(1e-7));

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(134.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(134.0)).epsilon(1e-7));

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(102.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(102.0)).epsilon(1e-7));

  // The other bound fully overlaps the bound.
  REQUIRE(b.MaxDistance(b) == Approx(sqrt(46.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(c) == Approx(sqrt(61.0)).epsilon(1e-7));

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  REQUIRE(b.MaxDistance(c) == Approx(sqrt(100.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(b) == Approx(sqrt(100.0)).epsilon(1e-7));

  // Identical bounds.  This will be the sum of the squared widths in each
  // dimension.
  REQUIRE(b.MaxDistance(b) == Approx(sqrt(46.0)).epsilon(1e-7));
  REQUIRE(c.MaxDistance(c) == Approx(sqrt(162.0)).epsilon(1e-7));

  // One last additional case.  If the bound encloses only one point, the
  // maximum distance between it and itself is 0.
  HRectBound<EuclideanDistance> d(2);

  d[0] = Range(2.0, 2.0);
  d[1] = Range(3.0, 3.0);

  REQUIRE(d.MaxDistance(d) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the behavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.
 */
TEST_CASE("HRectBoundRootRangeDistanceBound", "[TreeTest]")
{
  for (int i = 0; i < 50; ++i)
  {
    size_t dim = math::RandInt(20);

    HRectBound<EuclideanDistance> a(dim);
    HRectBound<EuclideanDistance> b(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec loA(dim);
    arma::vec widthA(dim);

    loA.randu();
    widthA.randu();

    arma::vec lo_b(dim);
    arma::vec width_b(dim);

    lo_b.randu();
    width_b.randu();

    for (size_t j = 0; j < dim; ++j)
    {
      a[j] = Range(loA[j], loA[j] + widthA[j]);
      b[j] = Range(lo_b[j], lo_b[j] + width_b[j]);
    }

    // Now ensure that MinDistance and MaxDistance report the same.
    Range r = a.RangeDistance(b);
    Range s = b.RangeDistance(a);

    REQUIRE(r.Lo() == Approx(s.Lo()).epsilon(1e-7));
    REQUIRE(r.Hi() == Approx(s.Hi()).epsilon(1e-7));

    REQUIRE(r.Lo() == Approx(a.MinDistance(b)).epsilon(1e-7));
    REQUIRE(r.Hi() == Approx(a.MaxDistance(b)).epsilon(1e-7));

    REQUIRE(s.Lo() == Approx(b.MinDistance(a)).epsilon(1e-7));
    REQUIRE(s.Hi() == Approx(b.MaxDistance(a)).epsilon(1e-7));
  }
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the bheavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.  This is for the
 * bound-to-point case.
 */
TEST_CASE("HRectBoundRootRangeDistancePoint", "[TreeTest]")
{
  for (int i = 0; i < 20; ++i)
  {
    size_t dim = math::RandInt(20);

    HRectBound<EuclideanDistance> a(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec loA(dim);
    arma::vec widthA(dim);

    loA.randu();
    widthA.randu();

    for (size_t j = 0; j < dim; ++j)
      a[j] = Range(loA[j], loA[j] + widthA[j]);

    // Now run the test on a few points.
    for (int j = 0; j < 10; ++j)
    {
      arma::vec point(dim);

      point.randu();

      Range r = a.RangeDistance(point);

      REQUIRE(r.Lo() == Approx(a.MinDistance(point)).epsilon(1e-7));
      REQUIRE(r.Hi() == Approx(a.MaxDistance(point)).epsilon(1e-7));
    }
  }
}

/**
 * Ensure that HRectBound::Diameter() works properly.
 */
TEST_CASE("HRectBoundDiameter", "[TreeTest]")
{
  HRectBound<LMetric<3, true>> b(4);
  b[0] = math::Range(0.0, 1.0);
  b[1] = math::Range(-1.0, 0.0);
  b[2] = math::Range(2.0, 3.0);
  b[3] = math::Range(7.0, 7.0);

  REQUIRE(b.Diameter()== Approx(std::pow(3.0, 1.0 / 3.0)).epsilon(1e-7));

  HRectBound<LMetric<2, false>> c(4);
  c[0] = math::Range(0.0, 1.0);
  c[1] = math::Range(-1.0, 0.0);
  c[2] = math::Range(2.0, 3.0);
  c[3] = math::Range(0.0, 0.0);

  REQUIRE(c.Diameter() == Approx(3.0).epsilon(1e-7));

  HRectBound<LMetric<5, true>> d(2);
  d[0] = math::Range(2.2, 2.2);
  d[1] = math::Range(1.0, 1.0);

  REQUIRE(d.Diameter() == Approx(0.0).margin(1e-5));
}

/**
 * It seems as though Bill has stumbled across a bug where
 * BinarySpaceTree<>::count() returns something different than
 * BinarySpaceTree<>::count_.  So, let's build a simple tree and make sure they
 * are the same.
 */
TEST_CASE("TreeCountMismatch", "[TreeTest]")
{
  arma::mat dataset = "2.0 5.0 9.0 4.0 8.0 7.0;"
                      "3.0 4.0 6.0 7.0 1.0 2.0 ";

  // Leaf size of 1.
  KDTree<EuclideanDistance, EmptyStatistic, arma::mat> rootNode(dataset, 1);

  REQUIRE(rootNode.Count() == 6);
  REQUIRE(rootNode.Left()->Count() == 3);
  REQUIRE(rootNode.Left()->Left()->Count() == 2);
  REQUIRE(rootNode.Left()->Left()->Left()->Count() == 1);
  REQUIRE(rootNode.Left()->Left()->Right()->Count() == 1);
  REQUIRE(rootNode.Left()->Right()->Count() == 1);
  REQUIRE(rootNode.Right()->Count() == 3);
  REQUIRE(rootNode.Right()->Left()->Count() == 2);
  REQUIRE(rootNode.Right()->Left()->Left()->Count() == 1);
  REQUIRE(rootNode.Right()->Left()->Right()->Count() == 1);
  REQUIRE(rootNode.Right()->Right()->Count() == 1);
}

TEST_CASE("CheckParents", "[TreeTest]")
{
  arma::mat dataset = "2.0 5.0 9.0 4.0 8.0 7.0;"
                      "3.0 4.0 6.0 7.0 1.0 2.0 ";

  // Leaf size of 1.
  KDTree<EuclideanDistance, EmptyStatistic, arma::mat> rootNode(dataset, 1);

  REQUIRE(rootNode.Parent() ==
      (KDTree<EuclideanDistance, EmptyStatistic, arma::mat>*) NULL);
  REQUIRE(&rootNode == rootNode.Left()->Parent());
  REQUIRE(&rootNode == rootNode.Right()->Parent());
  REQUIRE(rootNode.Left() == rootNode.Left()->Left()->Parent());
  REQUIRE(rootNode.Left() == rootNode.Left()->Right()->Parent());
  REQUIRE(rootNode.Left()->Left() ==
      rootNode.Left()->Left()->Left()->Parent());
  REQUIRE(rootNode.Left()->Left() ==
      rootNode.Left()->Left()->Right()->Parent());
  REQUIRE(rootNode.Right() == rootNode.Right()->Left()->Parent());
  REQUIRE(rootNode.Right() == rootNode.Right()->Right()->Parent());
  REQUIRE(rootNode.Right()->Left() ==
      rootNode.Right()->Left()->Left()->Parent());
  REQUIRE(rootNode.Right()->Left() ==
      rootNode.Right()->Left()->Right()->Parent());
}

TEST_CASE("CheckDataset", "[TreeTest]")
{
  arma::mat dataset = "2.0 5.0 9.0 4.0 8.0 7.0;"
                      "3.0 4.0 6.0 7.0 1.0 2.0 ";

  // Leaf size of 1.
  KDTree<EuclideanDistance, EmptyStatistic, arma::mat> rootNode(dataset, 1);

  arma::mat* rootDataset = &rootNode.Dataset();
  REQUIRE(&rootNode.Left()->Dataset() == rootDataset);
  REQUIRE(&rootNode.Right()->Dataset() == rootDataset);
  REQUIRE(&rootNode.Left()->Left()->Dataset() == rootDataset);
  REQUIRE(&rootNode.Left()->Right()->Dataset() == rootDataset);
  REQUIRE(&rootNode.Right()->Left()->Dataset() == rootDataset);
  REQUIRE(&rootNode.Right()->Right()->Dataset() == rootDataset);
  REQUIRE(&rootNode.Left()->Left()->Left()->Dataset() ==
      rootDataset);
  REQUIRE(&rootNode.Left()->Left()->Right()->Dataset() ==
      rootDataset);
  REQUIRE(&rootNode.Right()->Left()->Left()->Dataset() ==
      rootDataset);
  REQUIRE(&rootNode.Right()->Left()->Right()->Dataset() ==
      rootDataset);
}

// Ensure FurthestDescendantDistance() works.
TEST_CASE("FurthestDescendantDistanceTest", "[TreeTest]")
{
  arma::mat dataset = "1; 3"; // One point.
  KDTree<EuclideanDistance, EmptyStatistic, arma::mat> rootNode(dataset, 1);

  REQUIRE(rootNode.FurthestDescendantDistance() == Approx(0.0).margin(1e-5));

  dataset = "1 -1; 1 -1"; // Square of size [2, 2].

  // Both points are contained in the one node.
  KDTree<EuclideanDistance, EmptyStatistic, arma::mat> twoPoint(dataset);
  REQUIRE(twoPoint.FurthestDescendantDistance() ==
      Approx(sqrt(2.0)).epsilon(1e-7));
}

// Ensure that FurthestPointDistance() works.
TEST_CASE("FurthestPointDistanceTest", "[TreeTest]")
{
  arma::mat dataset;
  dataset.randu(5, 100);

  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(dataset);

  // Now, check each node.
  std::queue<TreeType*> nodeQueue;
  nodeQueue.push(&tree);

  while (!nodeQueue.empty())
  {
    TreeType* node = nodeQueue.front();
    nodeQueue.pop();

    if (node->NumChildren() != 0)
      REQUIRE(node->FurthestPointDistance() == 0.0);
    else
    {
      // Get center.
      arma::vec center;
      node->Center(center);

      double maxDist = 0.0;
      for (size_t i = 0; i < node->NumPoints(); ++i)
      {
        const double dist = metric::EuclideanDistance::Evaluate(center,
            dataset.col(node->Point(i)));
        if (dist > maxDist)
          maxDist = dist;
      }

      // We don't require an exact value because FurthestPointDistance() can
      // just bound the value instead of returning the exact value.
      REQUIRE(maxDist <= node->FurthestPointDistance());

      if (node->Left())
        nodeQueue.push(node->Left());
      if (node->Right())
        nodeQueue.push(node->Right());
    }
  }
}

TEST_CASE("ParentDistanceTest", "[TreeTest]")
{
  arma::mat dataset;
  dataset.randu(5, 500);

  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(dataset);

  // The root's parent distance should be 0 (although maybe it doesn't actually
  // matter; I just want to be sure it's not an uninitialized value, which this
  // test *sort* of checks).
  REQUIRE(tree.ParentDistance() == 0.0);

  // Do a depth-first traversal and make sure the parent distance is the same as
  // we calculate.
  std::stack<TreeType*> nodeStack;
  nodeStack.push(&tree);

  while (!nodeStack.empty())
  {
    TreeType* node = nodeStack.top();
    nodeStack.pop();

    // If it's a leaf, nothing to check.
    if (node->NumChildren() == 0)
      continue;

    arma::vec center, leftCenter, rightCenter;
    node->Center(center);
    node->Left()->Center(leftCenter);
    node->Right()->Center(rightCenter);

    const double leftDistance = LMetric<2>::Evaluate(center, leftCenter);
    const double rightDistance = LMetric<2>::Evaluate(center, rightCenter);

    REQUIRE(leftDistance ==
        Approx(node->Left()->ParentDistance()).epsilon(1e-7));
    REQUIRE(rightDistance ==
        Approx(node->Right()->ParentDistance()).epsilon(1e-7));

    nodeStack.push(node->Left());
    nodeStack.push(node->Right());
  }
}

TEST_CASE("ParentDistanceTestWithMapping", "[TreeTest]")
{
  arma::mat dataset;
  dataset.randu(5, 500);
  std::vector<size_t> oldFromNew;

  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(dataset, oldFromNew);

  // The root's parent distance should be 0 (although maybe it doesn't actually
  // matter; I just want to be sure it's not an uninitialized value, which this
  // test *sort* of checks).
  REQUIRE(tree.ParentDistance() == 0.0);

  // Do a depth-first traversal and make sure the parent distance is the same as
  // we calculate.
  std::stack<TreeType*> nodeStack;
  nodeStack.push(&tree);

  while (!nodeStack.empty())
  {
    TreeType* node = nodeStack.top();
    nodeStack.pop();

    // If it's a leaf, nothing to check.
    if (node->NumChildren() == 0)
      continue;

    arma::vec center, leftCenter, rightCenter;
    node->Center(center);
    node->Left()->Center(leftCenter);
    node->Right()->Center(rightCenter);

    const double leftDistance = LMetric<2>::Evaluate(center, leftCenter);
    const double rightDistance = LMetric<2>::Evaluate(center, rightCenter);

    REQUIRE(leftDistance ==
        Approx(node->Left()->ParentDistance()).epsilon(1e-7));
    REQUIRE(rightDistance ==
        Approx(node->Right()->ParentDistance()).epsilon(1e-7));

    nodeStack.push(node->Left());
    nodeStack.push(node->Right());
  }
}

// Forward declaration of methods we need for the next test.
template<typename TreeType>
bool CheckPointBounds(TreeType& node);

template<typename TreeType>
void GenerateVectorOfTree(TreeType* node,
                          size_t depth,
                          std::vector<TreeType*>& v);

/**
 * Exhaustive kd-tree test based on #125.
 *
 * - Generate a random dataset of a random size.
 * - Build a tree on that dataset.
 * - Ensure all the permutation indices map back to the correct points.
 * - Verify that each point is contained inside all of the bounds of its parent
 *     nodes.
 * - Verify that each bound at a particular level of the tree does not overlap
 *     with any other bounds at that level.
 *
 * Then, we do that whole process a handful of times.
 */
TEST_CASE("KdTreeTest", "[TreeTest]")
{
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  size_t maxRuns = 10; // Ten total tests.
  size_t pointIncrements = 1000; // Range is from 2000 points to 11000.

  // We use the default leaf size of 20.
  for (size_t run = 0; run < maxRuns; run++)
  {
    size_t dimensions = run + 2;
    size_t maxPoints = (run + 1) * pointIncrements;

    size_t size = maxPoints;
    arma::mat dataset = arma::mat(dimensions, size);

    // Mappings for post-sort verification of data.
    std::vector<size_t> newToOld;
    std::vector<size_t> oldToNew;

    // Generate data.
    dataset.randu();

    // Build the tree itself.
    TreeType root(dataset, newToOld, oldToNew);
    const arma::mat& treeset = root.Dataset();

    // Ensure the size of the tree is correct.
    REQUIRE(root.Count() == size);

    // Check the forward and backward mappings for correctness.
    for (size_t i = 0; i < size; ++i)
    {
      for (size_t j = 0; j < dimensions; ++j)
      {
        REQUIRE(treeset(j, i) == dataset(j, newToOld[i]));
        REQUIRE(treeset(j, oldToNew[i]) == dataset(j, i));
      }
    }

    // Now check that each point is contained inside of all bounds above it.
    CheckPointBounds(root);

    // Now check that no peers overlap.
    std::vector<TreeType*> v;
    GenerateVectorOfTree(&root, 1, v);

    // Start with the first pair.
    size_t depth = 2;
    // Compare each peer against every other peer.
    while (depth < v.size())
    {
      for (size_t i = depth; i < 2 * depth && i < v.size(); ++i)
        for (size_t j = i + 1; j < 2 * depth && j < v.size(); ++j)
          if (v[i] != NULL && v[j] != NULL)
            REQUIRE(!v[i]->Bound().Contains(v[j]->Bound()));

      depth *= 2;
    }
  }

  arma::mat dataset(25, 1000);
  for (size_t col = 0; col < dataset.n_cols; ++col)
    for (size_t row = 0; row < dataset.n_rows; ++row)
      dataset(row, col) = row + col;

  TreeType root(dataset);
}

TEST_CASE("MaxRPTreeTest", "[TreeTest]")
{
  typedef MaxRPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  size_t maxRuns = 10; // Ten total tests.
  size_t pointIncrements = 1000; // Range is from 2000 points to 11000.

  // We use the default leaf size of 20.
  for (size_t run = 0; run < maxRuns; run++)
  {
    size_t dimensions = run + 2;
    size_t maxPoints = (run + 1) * pointIncrements;

    size_t size = maxPoints;
    arma::mat dataset = arma::mat(dimensions, size);

    // Mappings for post-sort verification of data.
    std::vector<size_t> newToOld;
    std::vector<size_t> oldToNew;

    // Generate data.
    dataset.randu();

    // Build the tree itself.
    TreeType root(dataset, newToOld, oldToNew);
    const arma::mat& treeset = root.Dataset();

    // Ensure the size of the tree is correct.
    REQUIRE(root.Count() == size);

    // Check the forward and backward mappings for correctness.
    for (size_t i = 0; i < size; ++i)
    {
      for (size_t j = 0; j < dimensions; ++j)
      {
        REQUIRE(treeset(j, i) == dataset(j, newToOld[i]));
        REQUIRE(treeset(j, oldToNew[i]) == dataset(j, i));
      }
    }
  }
}

template<typename TreeType>
bool CheckHyperplaneSplit(const TreeType& tree)
{
  typedef typename TreeType::ElemType ElemType;

  const typename TreeType::Mat& dataset = tree.Dataset();
  arma::Mat<typename TreeType::ElemType> mat(dataset.n_rows + 1,
      tree.Left()->NumDescendants() + tree.Right()->NumDescendants());

  // We will try to find a hyperplane that splits the node.
  // The hyperplane may be represented as
  // a_1 * x_1 + ... + a_n * x_n + a_{n + 1} = 0.
  // We have to solve the system of inequalities (mat^t) * x <= 0,
  // where x[0], ... , x[dataset.n_rows-1] are the components of the normal
  // to the hyperplane and x[dataset.n_rows] is the position of the hyperplane
  // i.e. x = (a_1, ... , a_{n + 1}).
  // Each column of the matrix consists of a point and 1.
  // In such a way, the inner product of a column and x is equal to the value
  // of the hyperplane expression.
  // The hyperplane splits the node if the expression takes on opposite
  // values on node's children.

  for (size_t i = 0; i < tree.Left()->NumDescendants(); ++i)
  {
    for (size_t k = 0; k < dataset.n_rows; ++k)
      mat(k, i) = - dataset(k, tree.Left()->Descendant(i));

    mat(dataset.n_rows, i) = -1;
  }

  for (size_t i = 0; i < tree.Right()->NumDescendants(); ++i)
  {
    for (size_t k = 0; k < dataset.n_rows; ++k)
      mat(k, i + tree.Left()->NumDescendants()) =
          dataset(k, tree.Right()->Descendant(i));

    mat(dataset.n_rows, i + tree.Left()->NumDescendants()) = 1;
  }

  arma::Col<ElemType> x(dataset.n_rows + 1);
  x.zeros();
  // Define an initial value.
  x[0] = 1.0;
  x[1] = -arma::mean(
      dataset.cols(tree.Begin(), tree.Begin() + tree.Count() - 1).row(0));

  const size_t numIters = 1000000;
  const ElemType delta = 1e-4;

  // We will solve the system using a simple gradient method.
  bool success = false;
  for (size_t it = 0; it < numIters; it++)
  {
    success = true;
    for (size_t k = 0; k < tree.Count(); ++k)
    {
      ElemType result = arma::dot(mat.col(k), x);
      if (result > 0)
      {
        x -= mat.col(k) * delta;
        success = false;
      }
    }

    // The norm of the direction shouldn't be equal to zero.
    if (arma::norm(x.rows(0, dataset.n_rows-1)) < 1e-8)
    {
      x[math::RandInt(0, dataset.n_rows)] = 1.0;
      success = false;
    }

    if (success)
      break;
  }

  return success;
}

template<typename TreeType>
void CheckMaxRPTreeSplit(const TreeType& tree)
{
  if (tree.IsLeaf())
    return;

  REQUIRE(CheckHyperplaneSplit(tree) == true);

  CheckMaxRPTreeSplit(*tree.Left());
  CheckMaxRPTreeSplit(*tree.Right());
}

TEST_CASE("MaxRPTreeSplitTest", "[TreeTest]")
{
  typedef MaxRPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  arma::mat dataset;
  dataset.randu(8, 1000);
  TreeType root(dataset);

  CheckMaxRPTreeSplit(root);
}

TEST_CASE("RPTreeTest", "[TreeTest]")
{
  typedef RPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  size_t maxRuns = 10; // Ten total tests.
  size_t pointIncrements = 1000; // Range is from 2000 points to 11000.

  // We use the default leaf size of 20.
  for (size_t run = 0; run < maxRuns; run++)
  {
    size_t dimensions = run + 2;
    size_t maxPoints = (run + 1) * pointIncrements;

    size_t size = maxPoints;
    arma::mat dataset = arma::mat(dimensions, size);

    // Mappings for post-sort verification of data.
    std::vector<size_t> newToOld;
    std::vector<size_t> oldToNew;

    // Generate data.
    dataset.randu();

    // Build the tree itself.
    TreeType root(dataset, newToOld, oldToNew);
    const arma::mat& treeset = root.Dataset();

    // Ensure the size of the tree is correct.
    REQUIRE(root.Count() == size);

    // Check the forward and backward mappings for correctness.
    for (size_t i = 0; i < size; ++i)
    {
      for (size_t j = 0; j < dimensions; ++j)
      {
        REQUIRE(treeset(j, i) == dataset(j, newToOld[i]));
        REQUIRE(treeset(j, oldToNew[i]) == dataset(j, i));
      }
    }
  }
}

template<typename TreeType, typename MetricType>
void CheckRPTreeSplit(const TreeType& tree)
{
  typedef typename TreeType::ElemType ElemType;
  if (tree.IsLeaf())
    return;

  if (!CheckHyperplaneSplit(tree))
  {
    // Check if that was mean split.
    arma::Col<ElemType> center;
    tree.Left()->Bound().Center(center);
    ElemType maxDist = 0;
    for (size_t k =0; k < tree.Left()->NumDescendants(); ++k)
    {
      ElemType dist = MetricType::Evaluate(center,
          tree.Dataset().col(tree.Left()->Descendant(k)));

      if (dist > maxDist)
        maxDist = dist;
    }

    for (size_t k =0; k < tree.Right()->NumDescendants(); ++k)
    {
      ElemType dist = MetricType::Evaluate(center,
          tree.Dataset().col(tree.Right()->Descendant(k)));

      REQUIRE(maxDist <= dist *
          (1.0 + 10.0 * std::numeric_limits<ElemType>::epsilon()));
    }
  }

  CheckRPTreeSplit<TreeType, MetricType>(*tree.Left());
  CheckRPTreeSplit<TreeType, MetricType>(*tree.Right());
}

TEST_CASE("RPTreeSplitTest", "[TreeTest]")
{
  typedef RPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  arma::mat dataset;
  dataset.randu(8, 1000);
  TreeType root(dataset);

  CheckRPTreeSplit<TreeType, EuclideanDistance>(root);
}

// Recursively checks that each node contains all points that it claims to have.
template<typename TreeType>
bool CheckPointBounds(TreeType& node)
{
  // Check that each point which this tree claims is actually inside the tree.
  for (size_t index = 0; index < node.NumDescendants(); index++)
    if (!node.Bound().Contains(node.Dataset().col(node.Descendant(index))))
      return false;

  bool result = true;
  for (size_t child = 0; child < node.NumChildren(); ++child)
    result &= CheckPointBounds(node.Child(child));
  return result;
}

/**
 * Exhaustive ball tree test based on #125.
 *
 * - Generate a random dataset of a random size.
 * - Build a tree on that dataset.
 * - Ensure all the permutation indices map back to the correct points.
 * - Verify that each point is contained inside all of the bounds of its parent
 *     nodes.
 *
 * Then, we do that whole process a handful of times.
 */
TEST_CASE("BallTreeTest", "[TreeTest]")
{
  typedef BallTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  size_t maxRuns = 10; // Ten total tests.
  size_t pointIncrements = 1000; // Range is from 2000 points to 11000.

  // We use the default leaf size of 20.
  for (size_t run = 0; run < maxRuns; run++)
  {
    size_t dimensions = run + 2;
    size_t maxPoints = (run + 1) * pointIncrements;

    size_t size = maxPoints;
    arma::mat dataset = arma::mat(dimensions, size);
    arma::mat datacopy; // Used to test mappings.

    // Mappings for post-sort verification of data.
    std::vector<size_t> newToOld;
    std::vector<size_t> oldToNew;

    // Generate data.
    dataset.randu();

    // Build the tree itself.
    TreeType root(dataset, newToOld, oldToNew);
    const arma::mat& treeset = root.Dataset();

    // Ensure the size of the tree is correct.
    REQUIRE(root.NumDescendants() == size);

    // Check the forward and backward mappings for correctness.
    for (size_t i = 0; i < size; ++i)
    {
      for (size_t j = 0; j < dimensions; ++j)
      {
        REQUIRE(treeset(j, i) == dataset(j, newToOld[i]));
        REQUIRE(treeset(j, oldToNew[i]) == dataset(j, i));
      }
    }

    // Now check that each point is contained inside of all bounds above it.
    CheckPointBounds(root);
  }
}

/**
 * Ensure that we can build a ball tree with a custom instantiated metric type.
 */
TEST_CASE("MahalanobisBallTreeTest", "[TreeTest]")
{
  arma::mat dataset(10, 1000, arma::fill::randu);
  arma::mat cov = arma::eye<arma::mat>(10, 10);
  cov(2, 2) = 2.0; // Just so it's not completely the identity matrix.
  MahalanobisDistance<> m(std::move(cov));

  typedef BallTree<MahalanobisDistance<>, EmptyStatistic, arma::mat> TreeType;

  TreeType tree(dataset);

  // As long as it built successfully, I am okay with that.
  REQUIRE(tree.NumDescendants() == 1000);

  // Also test when we give oldFromNew, since this uses a different code path.
  std::vector<size_t> oldFromNew;
  TreeType tree2(std::move(dataset), oldFromNew);

  REQUIRE(tree.NumDescendants() == 1000);
}

template<typename TreeType>
void GenerateVectorOfTree(TreeType* node,
                          size_t depth,
                          std::vector<TreeType*>& v)
{
  if (node == NULL)
    return;

  if (depth >= v.size())
    v.resize(2 * depth + 1, NULL); // Resize to right size; fill with NULL.

  v[depth] = node;

  // Recurse to the left and right children.
  GenerateVectorOfTree(node->Left(), depth * 2, v);
  GenerateVectorOfTree(node->Right(), depth * 2 + 1, v);

  return;
}

/**
 * Exhaustive sparse kd-tree test based on #125.
 *
 * - Generate a random dataset of a random size.
 * - Build a tree on that dataset.
 * - Ensure all the permutation indices map back to the correct points.
 * - Verify that each point is contained inside all of the bounds of its parent
 *     nodes.
 * - Verify that each bound at a particular level of the tree does not overlap
 *     with any other bounds at that level.
 *
 * Then, we do that whole process a handful of times.
 */
TEST_CASE("ExhaustiveSparseKDTreeTest", "[TreeTest]")
{
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::SpMat<double>>
      TreeType;

  size_t maxRuns = 2; // Two total tests.
  size_t pointIncrements = 200; // Range is from 200 points to 400.

  // We use the default leaf size of 20.
  for (size_t run = 0; run < maxRuns; run++)
  {
    size_t dimensions = run + 2;
    size_t maxPoints = (run + 1) * pointIncrements;

    size_t size = maxPoints;
    arma::SpMat<double> dataset = arma::SpMat<double>(dimensions, size);
    arma::SpMat<double> datacopy; // Used to test mappings.

    // Mappings for post-sort verification of data.
    std::vector<size_t> newToOld;
    std::vector<size_t> oldToNew;

    // Generate data.
    dataset.sprandu(dimensions, size, 0.1);
    datacopy = dataset; // Save a copy.

    // Build the tree itself.
    TreeType root(dataset, newToOld, oldToNew);
    const arma::sp_mat& treeset = root.Dataset();

    // Ensure the size of the tree is correct.
    REQUIRE(root.Count() == size);

    // Check the forward and backward mappings for correctness.
    for (size_t i = 0; i < size; ++i)
    {
      for (size_t j = 0; j < dimensions; ++j)
      {
        REQUIRE(treeset(j, i) == dataset(j, newToOld[i]));
        REQUIRE(treeset(j, oldToNew[i]) == dataset(j, i));
      }
    }

    // Now check that each point is contained inside of all bounds above it.
    CheckPointBounds(root);

    // Now check that no peers overlap.
    std::vector<TreeType*> v;
    GenerateVectorOfTree(&root, 1, v);

    // Start with the first pair.
    size_t depth = 2;
    // Compare each peer against every other peer.
    while (depth < v.size())
    {
      for (size_t i = depth; i < 2 * depth && i < v.size(); ++i)
        for (size_t j = i + 1; j < 2 * depth && j < v.size(); ++j)
          if (v[i] != NULL && v[j] != NULL)
            REQUIRE(!v[i]->Bound().Contains(v[j]->Bound()));

      depth *= 2;
    }
  }

  arma::SpMat<double> dataset(25, 1000);
  for (size_t col = 0; col < dataset.n_cols; ++col)
    for (size_t row = 0; row < dataset.n_rows; ++row)
      dataset(row, col) = row + col;

  TreeType root(dataset);
}

TEST_CASE("BinarySpaceTreeMoveConstructorTest", "[TreeTest]")
{
  arma::mat dataset(5, 1000);
  dataset.randu();

  BinarySpaceTree<EuclideanDistance> tree(dataset);
  BinarySpaceTree<EuclideanDistance> tree2(std::move(tree));

  REQUIRE(tree.NumChildren() == 0);
  REQUIRE(tree2.NumChildren() == 2);
}

template<typename TreeType>
void RecurseTreeCountLeaves(const TreeType& node, arma::vec& counts)
{
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    if (node.Child(i).NumChildren() == 0)
      counts[node.Child(i).Point()]++;
    else
      RecurseTreeCountLeaves<TreeType>(node.Child(i), counts);
  }
}

template<typename TreeType>
void CheckSelfChild(const TreeType& node)
{
  if (node.NumChildren() == 0)
    return; // No self-child applicable here.

  bool found = false;
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    if (node.Child(i).Point() == node.Point())
      found = true;

    // Recursively check the children.
    CheckSelfChild(node.Child(i));
  }

  // Ensure this has its own self-child.
  REQUIRE(found == true);
}

template<typename TreeType, typename MetricType>
void CheckCovering(const TreeType& node)
{
  // Return if a leaf.  No checking necessary.
  if (node.NumChildren() == 0)
    return;

  const typename TreeType::Mat& dataset = node.Dataset();
  const size_t nodePoint = node.Point();

  // To ensure that this node satisfies the covering principle, we must ensure
  // that the distance to each child is less than pow(base, scale).
  double maxDistance = pow(node.Base(), node.Scale());
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    const size_t childPoint = node.Child(i).Point();

    double distance = MetricType::Evaluate(dataset.col(nodePoint),
        dataset.col(childPoint));

    REQUIRE(distance <= maxDistance);

    // Check the child.
    CheckCovering<TreeType, MetricType>(node.Child(i));
  }
}

/**
 * Create a simple cover tree and then make sure it is valid.
 */
TEST_CASE("SimpleCoverTreeConstructionTest", "[TreeTest]")
{
  // 20-point dataset.
  arma::mat data = arma::trans(arma::mat("0.0 0.0;"
                                         "1.0 0.0;"
                                         "0.5 0.5;"
                                         "2.0 2.0;"
                                         "-1.0 2.0;"
                                         "3.0 0.0;"
                                         "1.5 5.5;"
                                         "-2.0 -2.0;"
                                         "-1.5 1.5;"
                                         "0.0 4.0;"
                                         "2.0 1.0;"
                                         "2.0 1.2;"
                                         "-3.0 -2.5;"
                                         "-5.0 -5.0;"
                                         "3.5 1.5;"
                                         "2.0 2.5;"
                                         "-1.0 -1.0;"
                                         "-3.5 1.5;"
                                         "3.5 -1.5;"
                                         "2.0 1.0;"));

  // The root point will be the first point, (0, 0).
  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType tree(data); // Expansion constant of 2.0.

  // The furthest point from the root will be (-5, -5), with a distance of
  // of sqrt(50).  This means the scale of the root node should be 3 (because
  // 2^3 = 8).
  REQUIRE(tree.Scale() == 3);

  // Now loop through the tree and ensure that each leaf is only created once.
  arma::vec counts;
  counts.zeros(20);
  RecurseTreeCountLeaves(tree, counts);

  // Each point should only have one leaf node representing it.
  for (size_t i = 0; i < 20; ++i)
    REQUIRE(counts[i] == 1);

  // Each non-leaf should have a self-child.
  CheckSelfChild<TreeType>(tree);

  // Each node must satisfy the covering principle (its children must be less
  // than or equal to a certain distance apart).
  CheckCovering<TreeType, LMetric<2, true>>(tree);

  // There's no need to check the separation invariant because that is relaxed
  // in our implementation.
}

/**
 * Create a large cover tree and make sure it's accurate.
 */
TEST_CASE("CoverTreeConstructionTest", "[TreeTest]")
{
  arma::mat dataset;
  // 50-dimensional, 1000 point.
  dataset.randu(50, 1000);

  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType tree(dataset);

  // Ensure each leaf is only created once.
  arma::vec counts;
  counts.zeros(1000);
  RecurseTreeCountLeaves(tree, counts);

  for (size_t i = 0; i < 1000; ++i)
    REQUIRE(counts[i] == 1);

  // Each non-leaf should have a self-child.
  CheckSelfChild<TreeType>(tree);

  // Each node must satisfy the covering principle (its children must be less
  // than or equal to a certain distance apart).
  CheckCovering<TreeType, LMetric<2, true> >(tree);

  // There's no need to check the separation because that is relaxed in our
  // implementation.
}

/**
 * Create a cover tree on sparse data and make sure it's accurate.
 */
TEST_CASE("SparseCoverTreeConstructionTest", "[TreeTest]")
{
  arma::sp_mat dataset;
  // 50-dimensional, 1000 point.
  dataset.sprandu(50, 1000, 0.3);

  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::sp_mat>
      TreeType;
  TreeType tree(dataset);

  // Ensure each leaf is only created once.
  arma::vec counts;
  counts.zeros(1000);
  RecurseTreeCountLeaves(tree, counts);

  for (size_t i = 0; i < 1000; ++i)
    REQUIRE(counts[i] == 1);

  // Each non-leaf should have a self-child.
  CheckSelfChild<TreeType>(tree);

  // Each node must satisfy the covering principle (its children must be less
  // than or equal to a certain distance apart).
  CheckCovering<TreeType, LMetric<2, true> >(tree);

  // There's no need to check the separation invariant because that is relaxed
  // in our implementation.
}

/**
 * Test the manual constructor.
 */
TEST_CASE("CoverTreeManualConstructorTest", "[TreeTest]")
{
  arma::mat dataset;
  dataset.zeros(10, 10);

  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType node(dataset, 1.3, 3, 2, NULL, 1.5, 2.75);

  REQUIRE(&node.Dataset() == &dataset);
  REQUIRE(node.Base() == 1.3);
  REQUIRE(node.Point() == 3);
  REQUIRE(node.Scale() == 2);
  REQUIRE(node.Parent() == (CoverTree<>*) NULL);
  REQUIRE(node.ParentDistance() == 1.5);
  REQUIRE(node.FurthestDescendantDistance() == 2.75);
}

/**
 * Make sure cover trees work in different metric spaces.
 */
TEST_CASE("CoverTreeAlternateMetricTest", "[TreeTest]")
{
  arma::mat dataset;
  // 5-dimensional, 300-point dataset.
  dataset.randu(5, 300);

  typedef StandardCoverTree<ManhattanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType tree(dataset);

  // Ensure each leaf is only created once.
  arma::vec counts;
  counts.zeros(300);
  RecurseTreeCountLeaves<TreeType>(tree, counts);

  for (size_t i = 0; i < 300; ++i)
    REQUIRE(counts[i] == 1);

  // Each non-leaf should have a self-child.
  CheckSelfChild<TreeType>(tree);

  // Each node must satisfy the covering principle (its children must be less
  // than or equal to a certain distance apart).
  CheckCovering<TreeType, ManhattanDistance>(tree);

  // There's no need to check the separation invariant because that is relaxed
  // in our implementation.
}

/**
 * Make sure copy constructor works for the cover tree.
 */
TEST_CASE("CoverTreeCopyConstructor", "[TreeTest]")
{
  arma::mat dataset;
  dataset.randu(10, 10); // dataset is irrelevant.
  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType c(dataset, 1.3, 0, 5, NULL, 1.45, 5.2); // Random parameters.
  c.Children().push_back(new TreeType(dataset, 1.3, 1, 4, &c, 1.3, 2.45));
  c.Children().push_back(new TreeType(dataset, 1.5, 2, 3, &c, 1.2, 5.67));

  TreeType d = c;

  // Check that everything is the same.
  // As the tree being copied doesn't own the dataset, they must share the same
  // pointer.
  REQUIRE(c.Dataset().memptr() == d.Dataset().memptr());
  REQUIRE(c.Base() == Approx(d.Base()).epsilon(1e-52));
  REQUIRE(c.Point() == d.Point());
  REQUIRE(c.Scale() == d.Scale());
  REQUIRE(c.Parent() == d.Parent());
  REQUIRE(c.ParentDistance() == d.ParentDistance());
  REQUIRE(c.FurthestDescendantDistance() ==
                      d.FurthestDescendantDistance());
  REQUIRE(c.NumChildren() == d.NumChildren());
  REQUIRE(&c.Child(0) != &d.Child(0));
  REQUIRE(&c.Child(1) != &d.Child(1));

  REQUIRE(c.Child(0).Parent() == &c);
  REQUIRE(c.Child(1).Parent() == &c);
  REQUIRE(d.Child(0).Parent() == &d);
  REQUIRE(d.Child(1).Parent() == &d);

  // Check that the children are okay.
  REQUIRE(c.Child(0).Dataset().memptr() == c.Dataset().memptr());
  REQUIRE(c.Child(0).Base() == Approx(d.Child(0).Base()).epsilon(1e-52));
  REQUIRE(c.Child(0).Point() == d.Child(0).Point());
  REQUIRE(c.Child(0).Scale() == d.Child(0).Scale());
  REQUIRE(c.Child(0).ParentDistance() == d.Child(0).ParentDistance());
  REQUIRE(c.Child(0).FurthestDescendantDistance() ==
                      d.Child(0).FurthestDescendantDistance());
  REQUIRE(c.Child(0).NumChildren() == d.Child(0).NumChildren());

  REQUIRE(c.Child(1).Dataset().memptr() == c.Dataset().memptr());
  REQUIRE(c.Child(1).Base() == Approx(d.Child(1).Base()).epsilon(1e-52));
  REQUIRE(c.Child(1).Point() == d.Child(1).Point());
  REQUIRE(c.Child(1).Scale() == d.Child(1).Scale());
  REQUIRE(c.Child(1).ParentDistance() == d.Child(1).ParentDistance());
  REQUIRE(c.Child(1).FurthestDescendantDistance() ==
                      d.Child(1).FurthestDescendantDistance());
  REQUIRE(c.Child(1).NumChildren() == d.Child(1).NumChildren());

  // Check copy constructor when the tree being copied owns the dataset.
  TreeType e(std::move(dataset), 1.3);
  TreeType f = e;
  // As the tree being copied owns the dataset, they must have different
  // instances.
  REQUIRE(e.Dataset().memptr() != f.Dataset().memptr());
}

TEST_CASE("CoverTreeMoveDatasetTest", "[TreeTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 1000);
  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;

  TreeType t(std::move(dataset));

  REQUIRE(dataset.n_elem == 0);
  REQUIRE(t.Dataset().n_rows == 3);
  REQUIRE(t.Dataset().n_cols == 1000);

  EuclideanDistance ed; // Test the other constructor.
  dataset = arma::randu<arma::mat>(3, 1000);
  TreeType t2(std::move(dataset), ed);

  REQUIRE(dataset.n_elem == 0);
  REQUIRE(t2.Dataset().n_rows == 3);
  REQUIRE(t2.Dataset().n_cols == 1000);
}

/**
 * Make sure copy constructor works right for the binary space tree.
 */
TEST_CASE("BinarySpaceTreeCopyConstructor", "[TreeTest]")
{
  arma::mat data("1");
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType b(data);
  b.Begin() = 10;
  b.Count() = 50;

  b.Left() = new TreeType(data);
  b.Left()->Begin() = 10;
  b.Left()->Count() = 30;
  b.Left()->Parent() = &b;
  b.Right() = new TreeType(data);
  b.Right()->Begin() = 40;
  b.Right()->Count() = 20;
  b.Right()->Parent() = &b;

  // Copy the tree.
  TreeType c(b);

  // Ensure everything copied correctly.
  REQUIRE(b.Begin() == c.Begin());
  REQUIRE(b.Count() == c.Count());
  REQUIRE(b.Left() != c.Left());
  REQUIRE(b.Right() != c.Right());

  // Check the children.
  REQUIRE(b.Left()->Begin() == c.Left()->Begin());
  REQUIRE(b.Left()->Count() == c.Left()->Count());
  REQUIRE(b.Left()->Left() == (TreeType*) NULL);
  REQUIRE(b.Left()->Left() == c.Left()->Left());
  REQUIRE(b.Left()->Right() == (TreeType*) NULL);
  REQUIRE(b.Left()->Right() == c.Left()->Right());

  REQUIRE(b.Right()->Begin() == c.Right()->Begin());
  REQUIRE(b.Right()->Count() == c.Right()->Count());
  REQUIRE(b.Right()->Left() == (TreeType*) NULL);
  REQUIRE(b.Right()->Left() == c.Right()->Left());
  REQUIRE(b.Right()->Right() == (TreeType*) NULL);
  REQUIRE(b.Right()->Right() == c.Right()->Right());

  // Clean memory (we built the tree by hand, so this is what we have to do
  // since the destructor won't free the children's datasets).
  delete &b.Left()->Dataset();
  delete &b.Right()->Dataset();
}

//! Count the number of leaves under this node.
template<typename TreeType>
size_t NumLeaves(TreeType* node)
{
  if (node->NumChildren() == 0)
    return 1;

  size_t count = 0;
  for (size_t i = 0; i < node->NumChildren(); ++i)
    count += NumLeaves(&node->Child(i));

  return count;
}

//! Returns true if the index is contained somewhere under this node.
template<typename TreeType>
bool FindIndex(TreeType* node, const size_t index)
{
  for (size_t i = 0; i < node->NumPoints(); ++i)
    if (node->Point(i) == index)
      return true;

  for (size_t i = 0; i < node->NumChildren(); ++i)
    if (FindIndex(&node->Child(i), index))
      return true;

  return false;
}

//! Check that the points in the given node are accessible through the
//! Descendant() function of the root node.
template<typename TreeType>
bool CheckAccessibility(TreeType* childNode, TreeType* rootNode)
{
  for (size_t i = 0; i < childNode->NumPoints(); ++i)
  {
    bool found = false;
    for (size_t j = 0; j < rootNode->NumDescendants(); ++j)
    {
      if (childNode->Point(i) == rootNode->Descendant(j))
      {
        found = true;
        break;
      }
    }

    if (!found)
    {
      Log::Debug << "Did not find descendant " << childNode->Point(i) << ".\n";
      return false;
    }
  }

  // Now check the children.
  for (size_t i = 0; i < childNode->NumChildren(); ++i)
    if (!CheckAccessibility(&childNode->Child(i), rootNode))
      return false;

  return true;
}

//! Check that Descendant() and NumDescendants() is right for this node.
template<typename TreeType>
void CheckDescendants(TreeType* node)
{
  // In a cover tree, the number of leaves should be the number of descendant
  // points.
  const size_t numLeaves = NumLeaves(node);
  REQUIRE(numLeaves == node->NumDescendants());

  // Now check that each descendant is somewhere in the tree.
  for (size_t i = 0; i < node->NumDescendants(); ++i)
  {
    Log::Debug << "Check for descendant " << node->Descendant(i) << " (i " <<
        i << ").\n";
    REQUIRE(FindIndex(node, node->Descendant(i)) == true);
  }

  // Now check that every actual descendant is accessible through the
  // Descendant() function.
  REQUIRE(CheckAccessibility(node, node) == true);

  // Now check that there are no duplicates in the list of descendants.
  std::vector<size_t> descendants;
  descendants.resize(node->NumDescendants());
  for (size_t i = 0; i < node->NumDescendants(); ++i)
    descendants[i] = node->Descendant(i);

  // Sort the list.
  std::sort(descendants.begin(), descendants.end());

  // Check that there are no duplicates (this is easy because it's sorted).
  for (size_t i = 1; i < descendants.size(); ++i)
    REQUIRE(descendants[i] != descendants[i - 1]);

  // Now perform these same checks for the children.
  for (size_t i = 0; i < node->NumChildren(); ++i)
    CheckDescendants(&node->Child(i));
}

/**
 * Make sure Descendant() and NumDescendants() works properly for the cover
 * tree.
 */
TEST_CASE("CoverTreeDescendantTest", "[TreeTest]")
{
  arma::mat dataset;
  dataset.randu(3, 100);

  StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat> tree(dataset);

  // Now check that the NumDescendants() count and each Descendant() is right
  // using the recursive function above.
  CheckDescendants(&tree);
}
