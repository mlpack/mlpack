/**
 * @file tree_test.cpp
 *
 * Tests for tree-building methods.
 *
 * This file is part of MLPACK 1.0.7.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree/binary_space_tree.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/cover_tree/cover_tree.hpp>
#include <mlpack/core/tree/cosine_tree/cosine_tree.hpp>
#include <mlpack/core/tree/cosine_tree/cosine_tree_builder.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(TreeTest);

/**
 * Ensure that a bound, by default, is empty and has no dimensionality.
 */
BOOST_AUTO_TEST_CASE(HRectBoundEmptyConstructor)
{
  HRectBound<2> b;

  BOOST_REQUIRE_EQUAL((int) b.Dim(), 0);
}

/**
 * Ensure that when we specify the dimensionality in the constructor, it is
 * correct, and the bounds are all the empty set.
 */
BOOST_AUTO_TEST_CASE(HRectBoundDimConstructor)
{
  HRectBound<2> b(2); // We'll do this with 2 and 5 dimensions.

  BOOST_REQUIRE_EQUAL(b.Dim(), 2);
  BOOST_REQUIRE_SMALL(b[0].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[1].Width(), 1e-5);

  b = HRectBound<2>(5);

  BOOST_REQUIRE_EQUAL(b.Dim(), 5);
  BOOST_REQUIRE_SMALL(b[0].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[1].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[2].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[3].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[4].Width(), 1e-5);
}

/**
 * Test the copy constructor.
 */
BOOST_AUTO_TEST_CASE(HRectBoundCopyConstructor)
{
  HRectBound<2> b(2);
  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 3.0);

  HRectBound<2> c(b);

  BOOST_REQUIRE_EQUAL(c.Dim(), 2);
  BOOST_REQUIRE_SMALL(c[0].Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(c[0].Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Lo(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Hi(), 3.0, 1e-5);
}

/**
 * Test the assignment operator.
 */
BOOST_AUTO_TEST_CASE(HRectBoundAssignmentOperator)
{
  HRectBound<2> b(2);
  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 3.0);

  HRectBound<2> c(4);

  c = b;

  BOOST_REQUIRE_EQUAL(c.Dim(), 2);
  BOOST_REQUIRE_SMALL(c[0].Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(c[0].Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Lo(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Hi(), 3.0, 1e-5);
}

/**
 * Test that clearing the dimensions resets the bound to empty.
 */
BOOST_AUTO_TEST_CASE(HRectBoundClear)
{
  HRectBound<2> b(2); // We'll do this with two dimensions only.

  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 4.0);

  // Now we just need to make sure that we clear the range.
  b.Clear();

  BOOST_REQUIRE_SMALL(b[0].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[1].Width(), 1e-5);
}

/**
 * Ensure that we get the correct centroid for our bound.
 */
BOOST_AUTO_TEST_CASE(HRectBoundCentroid)
{
  // Create a simple 3-dimensional bound.
  HRectBound<2> b(3);

  b[0] = Range(0.0, 5.0);
  b[1] = Range(-2.0, -1.0);
  b[2] = Range(-10.0, 50.0);

  arma::vec centroid;

  b.Centroid(centroid);

  BOOST_REQUIRE_EQUAL(centroid.n_elem, 3);
  BOOST_REQUIRE_CLOSE(centroid[0], 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE(centroid[1], -1.5, 1e-5);
  BOOST_REQUIRE_CLOSE(centroid[2], 20.0, 1e-5);
}

/**
 * Ensure that we calculate the correct minimum distance between a point and a
 * bound.
 */
BOOST_AUTO_TEST_CASE(HRectBoundMinDistancePoint)
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<2> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  BOOST_REQUIRE_CLOSE(b.MinDistance(point), sqrt(95.0), 1e-5);

  point = "2.0 5.0 2.0 -5.0 1.0";

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);

  point = "1.0 2.0 0.0 -2.0 1.5";

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);
}

/**
 * Ensure that we calculate the correct minimum distance between a bound and
 * another bound.
 */
BOOST_AUTO_TEST_CASE(HRectBoundMinDistanceBound)
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<2> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<2> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  BOOST_REQUIRE_CLOSE(b.MinDistance(c), sqrt(22.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MinDistance(b), sqrt(22.0), 1e-5);

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(b), 1e-5);

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(b), 1e-5);

  // The other bound fully overlaps the bound.
  BOOST_REQUIRE_SMALL(b.MinDistance(b), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(c), 1e-5);

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(b), 1e-5);

  // Now we must be sure that the minimum distance to itself is 0.
  BOOST_REQUIRE_SMALL(b.MinDistance(b), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(c), 1e-5);
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and a
 * point.  This uses the same test cases as the MinDistance test.
 */
BOOST_AUTO_TEST_CASE(HRectBoundMaxDistancePoint)
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<2> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), sqrt(253.0), 1e-5);

  point = "2.0 5.0 2.0 -5.0 1.0";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), sqrt(46.0), 1e-5);

  point = "1.0 2.0 0.0 -2.0 1.5";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), sqrt(23.25), 1e-5);
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and
 * another bound.  This uses the same test cases as the MinDistance test.
 */
BOOST_AUTO_TEST_CASE(HRectBoundMaxDistanceBound)
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<2> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<2> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(210.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(210.0), 1e-5);

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(134.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(134.0), 1e-5);

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(102.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(102.0), 1e-5);

  // The other bound fully overlaps the bound.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(b), sqrt(46.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(c), sqrt(61.0), 1e-5);

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(100.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(100.0), 1e-5);

  // Identical bounds.  This will be the sum of the squared widths in each
  // dimension.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(b), sqrt(46.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(c), sqrt(162.0), 1e-5);

  // One last additional case.  If the bound encloses only one point, the
  // maximum distance between it and itself is 0.
  HRectBound<2> d(2);

  d[0] = Range(2.0, 2.0);
  d[1] = Range(3.0, 3.0);

  BOOST_REQUIRE_SMALL(d.MaxDistance(d), 1e-5);
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the behavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRangeDistanceBound)
{
  for (int i = 0; i < 50; i++)
  {
    size_t dim = math::RandInt(20);

    HRectBound<2> a(dim);
    HRectBound<2> b(dim);

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

    for (size_t j = 0; j < dim; j++)
    {
      a[j] = Range(loA[j], loA[j] + widthA[j]);
      b[j] = Range(lo_b[j], lo_b[j] + width_b[j]);
    }

    // Now ensure that MinDistance and MaxDistance report the same.
    Range r = a.RangeDistance(b);
    Range s = b.RangeDistance(a);

    BOOST_REQUIRE_CLOSE(r.Lo(), s.Lo(), 1e-5);
    BOOST_REQUIRE_CLOSE(r.Hi(), s.Hi(), 1e-5);

    BOOST_REQUIRE_CLOSE(r.Lo(), a.MinDistance(b), 1e-5);
    BOOST_REQUIRE_CLOSE(r.Hi(), a.MaxDistance(b), 1e-5);

    BOOST_REQUIRE_CLOSE(s.Lo(), b.MinDistance(a), 1e-5);
    BOOST_REQUIRE_CLOSE(s.Hi(), b.MaxDistance(a), 1e-5);
  }
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the bheavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.  This is for the
 * bound-to-point case.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRangeDistancePoint)
{
  for (int i = 0; i < 20; i++)
  {
    size_t dim = math::RandInt(20);

    HRectBound<2> a(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec loA(dim);
    arma::vec widthA(dim);

    loA.randu();
    widthA.randu();

    for (size_t j = 0; j < dim; j++)
      a[j] = Range(loA[j], loA[j] + widthA[j]);

    // Now run the test on a few points.
    for (int j = 0; j < 10; j++)
    {
      arma::vec point(dim);

      point.randu();

      Range r = a.RangeDistance(point);

      BOOST_REQUIRE_CLOSE(r.Lo(), a.MinDistance(point), 1e-5);
      BOOST_REQUIRE_CLOSE(r.Hi(), a.MaxDistance(point), 1e-5);
    }
  }
}

/**
 * Test that we can expand the bound to include a new point.
 */
BOOST_AUTO_TEST_CASE(HRectBoundOrOperatorPoint)
{
  // Because this should be independent in each dimension, we can essentially
  // run five test cases at once.
  HRectBound<2> b(5);

  b[0] = Range(1.0, 3.0);
  b[1] = Range(2.0, 4.0);
  b[2] = Range(-2.0, -1.0);
  b[3] = Range(0.0, 0.0);
  b[4] = Range(); // Empty range.

  arma::vec point = "2.0 4.0 2.0 -1.0 6.0";

  b |= point;

  BOOST_REQUIRE_CLOSE(b[0].Lo(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[0].Hi(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[1].Lo(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[1].Hi(), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[2].Lo(), -2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[2].Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[3].Lo(), -1.0, 1e-5);
  BOOST_REQUIRE_SMALL(b[3].Hi(), 1e-5);
  BOOST_REQUIRE_CLOSE(b[4].Lo(), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[4].Hi(), 6.0, 1e-5);
}

/**
 * Test that we can expand the bound to include another bound.
 */
BOOST_AUTO_TEST_CASE(HRectBoundOrOperatorBound)
{
  // Because this should be independent in each dimension, we can run many tests
  // at once.
  HRectBound<2> b(8);

  b[0] = Range(1.0, 3.0);
  b[1] = Range(2.0, 4.0);
  b[2] = Range(-2.0, -1.0);
  b[3] = Range(4.0, 5.0);
  b[4] = Range(2.0, 4.0);
  b[5] = Range(0.0, 0.0);
  b[6] = Range();
  b[7] = Range(1.0, 3.0);

  HRectBound<2> c(8);

  c[0] = Range(-3.0, -1.0); // Entirely less than the other bound.
  c[1] = Range(0.0, 2.0); // Touching edges.
  c[2] = Range(-3.0, -1.5); // Partially overlapping.
  c[3] = Range(4.0, 5.0); // Identical.
  c[4] = Range(1.0, 5.0); // Entirely enclosing.
  c[5] = Range(2.0, 2.0); // A single point.
  c[6] = Range(1.0, 3.0);
  c[7] = Range(); // Empty set.

  HRectBound<2> d = c;

  b |= c;
  d |= b;

  BOOST_REQUIRE_CLOSE(b[0].Lo(), -3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[0].Hi(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[0].Lo(), -3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[0].Hi(), 3.0, 1e-5);

  BOOST_REQUIRE_CLOSE(b[1].Lo(), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[1].Hi(), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[1].Lo(), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[1].Hi(), 4.0, 1e-5);

  BOOST_REQUIRE_CLOSE(b[2].Lo(), -3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[2].Hi(), -1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[2].Lo(), -3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[2].Hi(), -1.0, 1e-5);

  BOOST_REQUIRE_CLOSE(b[3].Lo(), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[3].Hi(), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[3].Lo(), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[3].Hi(), 5.0, 1e-5);

  BOOST_REQUIRE_CLOSE(b[4].Lo(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[4].Hi(), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[4].Lo(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[4].Hi(), 5.0, 1e-5);

  BOOST_REQUIRE_SMALL(b[5].Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(b[5].Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_SMALL(d[5].Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(d[5].Hi(), 2.0, 1e-5);

  BOOST_REQUIRE_CLOSE(b[6].Lo(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[6].Hi(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[6].Lo(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[6].Hi(), 3.0, 1e-5);

  BOOST_REQUIRE_CLOSE(b[7].Lo(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b[7].Hi(), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[7].Lo(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d[7].Hi(), 3.0, 1e-5);
}

/**
 * Test that the Contains() function correctly figures out whether or not a
 * point is in a bound.
 */
BOOST_AUTO_TEST_CASE(HRectBoundContains)
{
  // We can test a couple different points: completely outside the bound,
  // adjacent in one dimension to the bound, adjacent in all dimensions to the
  // bound, and inside the bound.
  HRectBound<2> b(3);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(0.0, 2.0);
  b[2] = Range(0.0, 2.0);

  // Completely outside the range.
  arma::vec point = "-1.0 4.0 4.0";
  BOOST_REQUIRE(!b.Contains(point));

  // Completely outside, but one dimension is in the range.
  point = "-1.0 4.0 1.0";
  BOOST_REQUIRE(!b.Contains(point));

  // Outside, but one dimension is on the edge.
  point = "-1.0 0.0 3.0";
  BOOST_REQUIRE(!b.Contains(point));

  // Two dimensions are on the edge, but one is outside.
  point = "0.0 0.0 3.0";
  BOOST_REQUIRE(!b.Contains(point));

  // Completely on the edge (should be contained).
  point = "0.0 0.0 0.0";
  BOOST_REQUIRE(b.Contains(point));

  // Inside the range.
  point = "0.3 1.0 0.4";
  BOOST_REQUIRE(b.Contains(point));
}

BOOST_AUTO_TEST_CASE(TestBallBound)
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

  BOOST_REQUIRE_CLOSE(b1.MinDistance(b2), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b1.RangeDistance(b2).Hi(), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b1.RangeDistance(b2).Lo(), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b1.RangeDistance(b2).Hi(), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b1.RangeDistance(b2).Lo(), 1-0.3-0.4, 1e-5);

  BOOST_REQUIRE_CLOSE(b2.MinDistance(b1), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b2.MaxDistance(b1), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b2.RangeDistance(b1).Hi(), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b2.RangeDistance(b1).Lo(), 1-0.3-0.4, 1e-5);

  BOOST_REQUIRE(b1.Contains(b1.Center()));
  BOOST_REQUIRE(!b1.Contains(b2.Center()));

  BOOST_REQUIRE(!b2.Contains(b1.Center()));
  BOOST_REQUIRE(b2.Contains(b2.Center()));
  arma::vec b2point(3); // A point that's within the radius but not the center.
  b2point[0] = 1.1;
  b2point[1] = 2.1;
  b2point[2] = 4.1;

  BOOST_REQUIRE(b2.Contains(b2point));

  BOOST_REQUIRE_SMALL(b1.MinDistance(b1.Center()), 1e-5);
  BOOST_REQUIRE_CLOSE(b1.MinDistance(b2.Center()), 1 - 0.3, 1e-5);
  BOOST_REQUIRE_CLOSE(b2.MinDistance(b1.Center()), 1 - 0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b2.MaxDistance(b1.Center()), 1 + 0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b1.MaxDistance(b2.Center()), 1 + 0.3, 1e-5);
}

/**
 * Ensure that we calculate the correct minimum distance between a point and a
 * bound.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRootMinDistancePoint)
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<2, true> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  BOOST_REQUIRE_CLOSE(b.MinDistance(point), sqrt(95.0), 1e-5);

  point = "2.0 5.0 2.0 -5.0 1.0";

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);

  point = "1.0 2.0 0.0 -2.0 1.5";

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);
}

/**
 * Ensure that we calculate the correct minimum distance between a bound and
 * another bound.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRootMinDistanceBound)
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<2, true> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<2, true> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  BOOST_REQUIRE_CLOSE(b.MinDistance(c), sqrt(22.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MinDistance(b), sqrt(22.0), 1e-5);

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(b), 1e-5);

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(b), 1e-5);

  // The other bound fully overlaps the bound.
  BOOST_REQUIRE_SMALL(b.MinDistance(b), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(c), 1e-5);

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(b), 1e-5);

  // Now we must be sure that the minimum distance to itself is 0.
  BOOST_REQUIRE_SMALL(b.MinDistance(b), 1e-5);
  BOOST_REQUIRE_SMALL(c.MinDistance(c), 1e-5);
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and a
 * point.  This uses the same test cases as the MinDistance test.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRootMaxDistancePoint)
{
  // We'll do the calculation in five dimensions, and we'll use three cases for
  // the point: point is outside the bound; point is on the edge of the bound;
  // point is inside the bound.  In the latter two cases, the distance should be
  // zero.
  HRectBound<2, true> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  arma::vec point = "-2.0 0.0 10.0 3.0 3.0";

  // This will be the Euclidean distance.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), sqrt(253.0), 1e-5);

  point = "2.0 5.0 2.0 -5.0 1.0";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), sqrt(46.0), 1e-5);

  point = "1.0 2.0 0.0 -2.0 1.5";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), sqrt(23.25), 1e-5);
}

/**
 * Ensure that we calculate the correct maximum distance between a bound and
 * another bound.  This uses the same test cases as the MinDistance test.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRootMaxDistanceBound)
{
  // We'll do the calculation in five dimensions, and we can use six cases.
  // The other bound is completely outside the bound; the other bound is on the
  // edge of the bound; the other bound partially overlaps the bound; the other
  // bound fully overlaps the bound; the other bound is entirely inside the
  // bound; the other bound entirely envelops the bound.
  HRectBound<2, true> b(5);

  b[0] = Range(0.0, 2.0);
  b[1] = Range(1.0, 5.0);
  b[2] = Range(-2.0, 2.0);
  b[3] = Range(-5.0, -2.0);
  b[4] = Range(1.0, 2.0);

  HRectBound<2, true> c(5);

  // The other bound is completely outside the bound.
  c[0] = Range(-5.0, -2.0);
  c[1] = Range(6.0, 7.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(2.0, 5.0);
  c[4] = Range(3.0, 4.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(210.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(210.0), 1e-5);

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(134.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(134.0), 1e-5);

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(102.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(102.0), 1e-5);

  // The other bound fully overlaps the bound.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(b), sqrt(46.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(c), sqrt(61.0), 1e-5);

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), sqrt(100.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), sqrt(100.0), 1e-5);

  // Identical bounds.  This will be the sum of the squared widths in each
  // dimension.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(b), sqrt(46.0), 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(c), sqrt(162.0), 1e-5);

  // One last additional case.  If the bound encloses only one point, the
  // maximum distance between it and itself is 0.
  HRectBound<2, true> d(2);

  d[0] = Range(2.0, 2.0);
  d[1] = Range(3.0, 3.0);

  BOOST_REQUIRE_SMALL(d.MaxDistance(d), 1e-5);
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the behavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRootRangeDistanceBound)
{
  for (int i = 0; i < 50; i++)
  {
    size_t dim = math::RandInt(20);

    HRectBound<2, true> a(dim);
    HRectBound<2, true> b(dim);

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

    for (size_t j = 0; j < dim; j++)
    {
      a[j] = Range(loA[j], loA[j] + widthA[j]);
      b[j] = Range(lo_b[j], lo_b[j] + width_b[j]);
    }

    // Now ensure that MinDistance and MaxDistance report the same.
    Range r = a.RangeDistance(b);
    Range s = b.RangeDistance(a);

    BOOST_REQUIRE_CLOSE(r.Lo(), s.Lo(), 1e-5);
    BOOST_REQUIRE_CLOSE(r.Hi(), s.Hi(), 1e-5);

    BOOST_REQUIRE_CLOSE(r.Lo(), a.MinDistance(b), 1e-5);
    BOOST_REQUIRE_CLOSE(r.Hi(), a.MaxDistance(b), 1e-5);

    BOOST_REQUIRE_CLOSE(s.Lo(), b.MinDistance(a), 1e-5);
    BOOST_REQUIRE_CLOSE(s.Hi(), b.MaxDistance(a), 1e-5);
  }
}

/**
 * Ensure that the ranges returned by RangeDistance() are equal to the minimum
 * and maximum distance.  We will perform this test by creating random bounds
 * and comparing the bheavior to MinDistance() and MaxDistance() -- so this test
 * is assuming that those passed and operate correctly.  This is for the
 * bound-to-point case.
 */
BOOST_AUTO_TEST_CASE(HRectBoundRootRangeDistancePoint)
{
  for (int i = 0; i < 20; i++)
  {
    size_t dim = math::RandInt(20);

    HRectBound<2, true> a(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec loA(dim);
    arma::vec widthA(dim);

    loA.randu();
    widthA.randu();

    for (size_t j = 0; j < dim; j++)
      a[j] = Range(loA[j], loA[j] + widthA[j]);

    // Now run the test on a few points.
    for (int j = 0; j < 10; j++)
    {
      arma::vec point(dim);

      point.randu();

      Range r = a.RangeDistance(point);

      BOOST_REQUIRE_CLOSE(r.Lo(), a.MinDistance(point), 1e-5);
      BOOST_REQUIRE_CLOSE(r.Hi(), a.MaxDistance(point), 1e-5);
    }
  }
}

/**
 * Ensure that HRectBound::Diameter() works properly.
 */
BOOST_AUTO_TEST_CASE(HRectBoundDiameter)
{
  HRectBound<3> b(4);
  b[0] = math::Range(0.0, 1.0);
  b[1] = math::Range(-1.0, 0.0);
  b[2] = math::Range(2.0, 3.0);
  b[3] = math::Range(7.0, 7.0);

  BOOST_REQUIRE_CLOSE(b.Diameter(), std::pow(3.0, 1.0 / 3.0), 1e-5);

  HRectBound<2, false> c(4);
  c[0] = math::Range(0.0, 1.0);
  c[1] = math::Range(-1.0, 0.0);
  c[2] = math::Range(2.0, 3.0);
  c[3] = math::Range(0.0, 0.0);

  BOOST_REQUIRE_CLOSE(c.Diameter(), 3.0, 1e-5);

  HRectBound<5> d(2);
  d[0] = math::Range(2.2, 2.2);
  d[1] = math::Range(1.0, 1.0);

  BOOST_REQUIRE_SMALL(d.Diameter(), 1e-5);
}

/**
 * Ensure that a bound, by default, is empty and has no dimensionality, and the
 * box size vector is empty.
 */
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundEmptyConstructor)
{
  PeriodicHRectBound<2> b;

  BOOST_REQUIRE_EQUAL(b.Dim(), 0);
  BOOST_REQUIRE_EQUAL(b.Box().n_elem, 0);
}

/**
 * Ensure that when we specify the dimensionality in the constructor, it is
 * correct, and the bounds are all the empty set.
 */
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundBoxConstructor)
{
  PeriodicHRectBound<2> b(arma::vec("5 6"));

  BOOST_REQUIRE_EQUAL(b.Dim(), 2);
  BOOST_REQUIRE_SMALL(b[0].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[1].Width(), 1e-5);
  BOOST_REQUIRE_EQUAL(b.Box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(b.Box()[0], 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b.Box()[1], 6.0, 1e-5);

  PeriodicHRectBound<2> d(arma::vec("2 3 4 5 6"));

  BOOST_REQUIRE_EQUAL(d.Dim(), 5);
  BOOST_REQUIRE_SMALL(d[0].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[1].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[2].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[3].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[4].Width(), 1e-5);
  BOOST_REQUIRE_EQUAL(d.Box().n_elem, 5);
  BOOST_REQUIRE_CLOSE(d.Box()[0], 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Box()[1], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Box()[2], 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Box()[3], 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Box()[4], 6.0, 1e-5);
}

/**
 * Test the copy constructor.
 */
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundCopyConstructor)
{
  PeriodicHRectBound<2> b(arma::vec("3 4"));
  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 3.0);

  PeriodicHRectBound<2> c(b);

  BOOST_REQUIRE_EQUAL(c.Dim(), 2);
  BOOST_REQUIRE_SMALL(c[0].Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(c[0].Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Lo(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Hi(), 3.0, 1e-5);
  BOOST_REQUIRE_EQUAL(c.Box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(c.Box()[0], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.Box()[1], 4.0, 1e-5);
}

/**
 * Test the assignment operator.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundAssignmentOperator)
{
  PeriodicHRectBound<2> b(arma::vec("3 4"));
  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 3.0);

  PeriodicHRectBound<2> c(arma::vec("3 4 5 6"));

  c = b;

  BOOST_REQUIRE_EQUAL(c.Dim(), 2);
  BOOST_REQUIRE_SMALL(c[0].Lo(), 1e-5);
  BOOST_REQUIRE_CLOSE(c[0].Hi(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Lo(), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c[1].Hi(), 3.0, 1e-5);
  BOOST_REQUIRE_EQUAL(c.Box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(c.Box()[0], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.Box()[1], 4.0, 1e-5);
}*/

/**
 * Ensure that we can set the box size correctly.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundSetBoxSize)
{
  PeriodicHRectBound<2> b(arma::vec("1 2"));

  b.SetBoxSize(arma::vec("10 12"));

  BOOST_REQUIRE_EQUAL(b.Box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(b.Box()[0], 10.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b.Box()[1], 12.0, 1e-5);
}*/

/**
 * Ensure that we can clear the dimensions correctly.  This does not involve the
 * box size at all, so the test can be identical to the HRectBound test.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundClear)
{
  // We'll do this with two dimensions only.
  PeriodicHRectBound<2> b(arma::vec("5 5"));

  b[0] = Range(0.0, 2.0);
  b[1] = Range(2.0, 4.0);

  // Now we just need to make sure that we clear the range.
  b.Clear();

  BOOST_REQUIRE_SMALL(b[0].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(b[1].Width(), 1e-5);
}*/

/**
 * Ensure that we get the correct centroid for our bound.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundCentroid) {
  // Create a simple 3-dimensional bound.  The centroid is not affected by the
  // periodic coordinates.
  PeriodicHRectBound<2> b(arma::vec("100 100 100"));

  b[0] = Range(0.0, 5.0);
  b[1] = Range(-2.0, -1.0);
  b[2] = Range(-10.0, 50.0);

  arma::vec centroid;

  b.Centroid(centroid);

  BOOST_REQUIRE_EQUAL(centroid.n_elem, 3);
  BOOST_REQUIRE_CLOSE(centroid[0], 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE(centroid[1], -1.5, 1e-5);
  BOOST_REQUIRE_CLOSE(centroid[2], 20.0, 1e-5);
}*/

/**
 * Correctly calculate the minimum distance between the bound and a point in
 * periodic coordinates.  We have to account for the shifts necessary in
 * periodic coordinates too, so that makes testing this a little more difficult.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundMinDistancePoint)
{
  // First, we'll start with a simple 2-dimensional case where the point is
  // inside the bound, then on the edge of the bound, then barely outside the
  // bound.  The box size will be large enough that this is basically the
  // HRectBound case.
  PeriodicHRectBound<2> b(arma::vec("100 100"));

  b[0] = Range(0.0, 5.0);
  b[1] = Range(2.0, 4.0);

  // Inside the bound.
  arma::vec point = "2.5 3.0";

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);

  // On the edge.
  point = "5.0 4.0";

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);

  // And just a little outside the bound.
  point = "6.0 5.0";

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 2.0, 1e-5);

  // Now we start to invoke the periodicity.  This point will "alias" to (-1,
  // -1).
  point = "99.0 99.0";

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 10.0, 1e-5);

  // We will perform several tests on a one-dimensional bound.
  PeriodicHRectBound<2> a(arma::vec("5.0"));
  point.set_size(1);

  a[0] = Range(2.0, 4.0); // Entirely inside box.
  point[0] = 7.5; // Inside first right image of the box.

  BOOST_REQUIRE_SMALL(a.MinDistance(point), 1e-5);

  a[0] = Range(0.0, 5.0); // Fills box fully.
  point[1] = 19.3; // Inside the box, which covers everything.

  BOOST_REQUIRE_SMALL(a.MinDistance(point), 1e-5);

  a[0] = Range(-10.0, 10.0); // Larger than the box.
  point[0] = -500.0; // Inside the box, which covers everything.

  BOOST_REQUIRE_SMALL(a.MinDistance(point), 1e-5);

  a[0] = Range(-2.0, 1.0); // Crosses over an edge.
  point[0] = 2.9; // The first right image of the bound starts at 3.0.

  BOOST_REQUIRE_CLOSE(a.MinDistance(point), 0.01, 1e-5);

  a[0] = Range(2.0, 4.0); // Inside box.
  point[0] = 0.0; // Closest to the first left image of the bound.

  BOOST_REQUIRE_CLOSE(a.MinDistance(point), 1.0, 1e-5);

  a[0] = Range(0.0, 2.0); // On edge of box.
  point[0] = 7.1; // 0.1 away from the first right image of the bound.

  BOOST_REQUIRE_CLOSE(a.MinDistance(point), 0.01, 1e-5);

  PeriodicHRectBound<2> d(arma::vec("0.0"));
  d[0] = Range(-10.0, 10.0); // Box is of infinite size.
  point[0] = 810.0; // 800 away from the only image of the box.

  BOOST_REQUIRE_CLOSE(d.MinDistance(point), 640000.0, 1e-5);

  PeriodicHRectBound<2> e(arma::vec("-5.0"));
  e[0] = Range(2.0, 4.0); // Box size of -5 should function the same as 5.
  point[0] = -10.8; // Should alias to 4.2.

  BOOST_REQUIRE_CLOSE(e.MinDistance(point), 0.04, 1e-5);

  // Switch our bound to a higher dimensionality.  This should ensure that the
  // dimensions are independent like they should be.
  PeriodicHRectBound<2> c(arma::vec("5.0 5.0 5.0 5.0 5.0 5.0 0.0 -5.0"));

  c[0] = Range(2.0, 4.0); // Entirely inside box.
  c[1] = Range(0.0, 5.0); // Fills box fully.
  c[2] = Range(-10.0, 10.0); // Larger than the box.
  c[3] = Range(-2.0, 1.0); // Crosses over an edge.
  c[4] = Range(2.0, 4.0); // Inside box.
  c[5] = Range(0.0, 2.0); // On edge of box.
  c[6] = Range(-10.0, 10.0); // Box is of infinite size.
  c[7] = Range(2.0, 4.0); // Box size of -5 should function the same as 5.

  point.set_size(8);
  point[0] = 7.5; // Inside first right image of the box.
  point[1] = 19.3; // Inside the box, which covers everything.
  point[2] = -500.0; // Inside the box, which covers everything.
  point[3] = 2.9; // The first right image of the bound starts at 3.0.
  point[4] = 0.0; // Closest to the first left image of the bound.
  point[5] = 7.1; // 0.1 away from the first right image of the bound.
  point[6] = 810.0; // 800 away from the only image of the box.
  point[7] = -10.8; // Should alias to 4.2.

  BOOST_REQUIRE_CLOSE(c.MinDistance(point), 640001.06, 1e-10);
}*/

/**
 * Correctly calculate the minimum distance between the bound and another bound in
 * periodic coordinates.  We have to account for the shifts necessary in
 * periodic coordinates too, so that makes testing this a little more difficult.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundMinDistanceBound)
{
  // First, we'll start with a simple 2-dimensional case where the bounds are nonoverlapping,
  // then one bound is on the edge of the other bound,
  //  then overlapping, then one range entirely covering the other.  The box size will be large enough that this is basically the
  // HRectBound case.
  PeriodicHRectBound<2> b(arma::vec("100 100"));
  PeriodicHRectBound<2> c(arma::vec("100 100"));

  b[0] = Range(0.0, 5.0);
  b[1] = Range(2.0, 4.0);

  // Inside the bound.
  c[0] = Range(7.0, 9.0);
  c[1] = Range(10.0,12.0);


  BOOST_REQUIRE_CLOSE(b.MinDistance(c), 40.0, 1e-5);

  // On the edge.
  c[0] = Range(5.0, 8.0);
  c[1] = Range(4.0, 6.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);

  // Overlapping the bound.
  c[0] = Range(3.0, 6.0);
  c[1] = Range(1.0, 3.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);

  // One range entirely covering the other

  c[0] = Range(0.0, 6.0);
  c[1] = Range(1.0, 7.0);

  BOOST_REQUIRE_SMALL(b.MinDistance(c), 1e-5);

  // Now we start to invoke the periodicity.  These bounds "alias" to (-3.0,
  // -1.0) and (5,0,6.0).

  c[0] = Range(97.0, 99.0);
  c[1] = Range(105.0, 106.0);

  BOOST_REQUIRE_CLOSE(b.MinDistance(c), 2.0, 1e-5);

  // We will perform several tests on a one-dimensional bound and smaller box size and mostly overlapping.
  PeriodicHRectBound<2> a(arma::vec("5.0"));
  PeriodicHRectBound<2> d(a);

  a[0] = Range(2.0, 4.0); // Entirely inside box.
  d[0] = Range(7.5, 10.0); // In the right image of the box, overlapping ranges.

  BOOST_REQUIRE_SMALL(a.MinDistance(d), 1e-5);

  a[0] = Range(0.0, 5.0); // Fills box fully.
  d[0] = Range(19.3, 21.0); // Two intervals inside the box, same as range of b[0].

  BOOST_REQUIRE_SMALL(a.MinDistance(d), 1e-5);

  a[0] = Range(-10.0, 10.0); // Larger than the box.
  d[0] = Range(-500.0, -498.0); // Inside the box, which covers everything.

  BOOST_REQUIRE_SMALL(a.MinDistance(d), 1e-5);

  a[0] = Range(-2.0, 1.0); // Crosses over an edge.
  d[0] = Range(2.9, 5.1); // The first right image of the bound starts at 3.0. Overlapping

  BOOST_REQUIRE_SMALL(a.MinDistance(d), 1e-5);

  a[0] = Range(-1.0, 1.0); // Crosses over an edge.
  d[0] = Range(11.9, 12.5); // The first right image of the bound starts at 4.0.
  BOOST_REQUIRE_CLOSE(a.MinDistance(d), 0.81, 1e-5);

  a[0] = Range(2.0, 3.0);
  d[0] = Range(9.5, 11);
  BOOST_REQUIRE_CLOSE(a.MinDistance(d), 1.0, 1e-5);

}*/

/**
 * Correctly calculate the maximum distance between the bound and a point in
 * periodic coordinates.  We have to account for the shifts necessary in
 * periodic coordinates too, so that makes testing this a little more difficult.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundMaxDistancePoint)
{
  // First, we'll start with a simple 2-dimensional case where the point is
  // inside the bound, then on the edge of the bound, then barely outside the
  // bound.  The box size will be large enough that this is basically the
  // HRectBound case.
  PeriodicHRectBound<2> b(arma::vec("100 100"));

  b[0] = Range(0.0, 5.0);
  b[1] = Range(2.0, 4.0);

  // Inside the bound.
  arma::vec point = "2.5 3.0";

  //BOOST_REQUIRE_CLOSE(b.MaxDistance(point), 7.25, 1e-5);

  // On the edge.
  point = "5.0 4.0";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), 29.0, 1e-5);

  // And just a little outside the bound.
  point = "6.0 5.0";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), 45.0, 1e-5);

  // Now we start to invoke the periodicity.  This point will "alias" to (-1,
  // -1).
  point = "99.0 99.0";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), 61.0, 1e-5);

  // We will perform several tests on a one-dimensional bound and smaller box size.
  PeriodicHRectBound<2> a(arma::vec("5.0"));
  point.set_size(1);

  a[0] = Range(2.0, 4.0); // Entirely inside box.
  point[0] = 7.5; // Inside first right image of the box.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(point), 2.25, 1e-5);

  a[0] = Range(0.0, 5.0); // Fills box fully.
  point[1] = 19.3; // Inside the box, which covers everything.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(point), 18.49, 1e-5);

  a[0] = Range(-10.0, 10.0); // Larger than the box.
  point[0] = -500.0; // Inside the box, which covers everything.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(point), 25.0, 1e-5);

  a[0] = Range(-2.0, 1.0); // Crosses over an edge.
  point[0] = 2.9; // The first right image of the bound starts at 3.0.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(point), 8.41, 1e-5);

  a[0] = Range(2.0, 4.0); // Inside box.
  point[0] = 0.0; // Farthest from the first right image of the bound.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(point), 25.0, 1e-5);

  a[0] = Range(0.0, 2.0); // On edge of box.
  point[0] = 7.1; // 2.1 away from the first left image of the bound.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(point), 4.41, 1e-5);

  PeriodicHRectBound<2> d(arma::vec("0.0"));
  d[0] = Range(-10.0, 10.0); // Box is of infinite size.
  point[0] = 810.0; // 820 away from the only left image of the box.

  BOOST_REQUIRE_CLOSE(d.MinDistance(point), 672400.0, 1e-5);

  PeriodicHRectBound<2> e(arma::vec("-5.0"));
  e[0] = Range(2.0, 4.0); // Box size of -5 should function the same as 5.
  point[0] = -10.8; // Should alias to 4.2.

  BOOST_REQUIRE_CLOSE(e.MaxDistance(point), 4.84, 1e-5);

  // Switch our bound to a higher dimensionality.  This should ensure that the
  // dimensions are independent like they should be.
  PeriodicHRectBound<2> c(arma::vec("5.0 5.0 5.0 5.0 5.0 5.0 0.0 -5.0"));

  c[0] = Range(2.0, 4.0); // Entirely inside box.
  c[1] = Range(0.0, 5.0); // Fills box fully.
  c[2] = Range(-10.0, 10.0); // Larger than the box.
  c[3] = Range(-2.0, 1.0); // Crosses over an edge.
  c[4] = Range(2.0, 4.0); // Inside box.
  c[5] = Range(0.0, 2.0); // On edge of box.
  c[6] = Range(-10.0, 10.0); // Box is of infinite size.
  c[7] = Range(2.0, 4.0); // Box size of -5 should function the same as 5.

  point.set_size(8);
  point[0] = 7.5; // Inside first right image of the box.
  point[1] = 19.3; // Inside the box, which covers everything.
  point[2] = -500.0; // Inside the box, which covers everything.
  point[3] = 2.9; // The first right image of the bound starts at 3.0.
  point[4] = 0.0; // Closest to the first left image of the bound.
  point[5] = 7.1; // 0.1 away from the first right image of the bound.
  point[6] = 810.0; // 800 away from the only image of the box.
  point[7] = -10.8; // Should alias to 4.2.

  BOOST_REQUIRE_CLOSE(c.MaxDistance(point), 672630.65, 1e-10);
}*/

/**
 * Correctly calculate the maximum distance between the bound and another bound in
 * periodic coordinates.  We have to account for the shifts necessary in
 * periodic coordinates too, so that makes testing this a little more difficult.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundMaxDistanceBound)
{
  // First, we'll start with a simple 2-dimensional case where the bounds are nonoverlapping,
  // then one bound is on the edge of the other bound,
  //  then overlapping, then one range entirely covering the other.  The box size will be large enough that this is basically the
  // HRectBound case.
  PeriodicHRectBound<2> b(arma::vec("100 100"));
  PeriodicHRectBound<2> c(b);

  b[0] = Range(0.0, 5.0);
  b[1] = Range(2.0, 4.0);

  // Inside the bound.

  c[0] = Range(7.0, 9.0);
  c[1] = Range(10.0,12.0);


  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 181.0, 1e-5);

  // On the edge.

  c[0] = Range(5.0, 8.0);
  c[1] = Range(4.0, 6.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 80.0, 1e-5);

  // Overlapping the bound.

  c[0] = Range(3.0, 6.0);
  c[1] = Range(1.0, 3.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 45.0, 1e-5);

  // One range entirely covering the other

  c[0] = Range(0.0, 6.0);
  c[1] = Range(1.0, 7.0);

   BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 61.0, 1e-5);

  // Now we start to invoke the periodicity.  Thess bounds "alias" to (-3.0,
  // -1.0) and (5,0,6.0).

  c[0] = Range(97.0, 99.0);
  c[1] = Range(105.0, 106.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 80.0, 1e-5);

  // We will perform several tests on a one-dimensional bound and smaller box size.
  PeriodicHRectBound<2> a(arma::vec("5.0"));
  PeriodicHRectBound<2> d(a);

  a[0] = Range(2.0, 4.0); // Entirely inside box.
  d[0] = Range(7.5, 10); // In the right image of the box, overlapping ranges.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(d), 9.0, 1e-5);

  a[0] = Range(0.0, 5.0); // Fills box fully.
  d[0] = Range(19.3, 21.0); // Two intervals inside the box, same as range of b[0].

  BOOST_REQUIRE_CLOSE(a.MaxDistance(d), 18.49, 1e-5);

  a[0] = Range(-10.0, 10.0); // Larger than the box.
  d[0] = Range(-500.0, -498.0); // Inside the box, which covers everything.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(d), 9.0, 1e-5);

  a[0] = Range(-2.0, 1.0); // Crosses over an edge.
  d[0] = Range(2.9, 5.1); // The first right image of the bound starts at 3.0.

  BOOST_REQUIRE_CLOSE(a.MaxDistance(d), 24.01, 1e-5);
}*/


/**
 * It seems as though Bill has stumbled across a bug where
 * BinarySpaceTree<>::count() returns something different than
 * BinarySpaceTree<>::count_.  So, let's build a simple tree and make sure they
 * are the same.
 */
BOOST_AUTO_TEST_CASE(TreeCountMismatch)
{
  arma::mat dataset = "2.0 5.0 9.0 4.0 8.0 7.0;"
                      "3.0 4.0 6.0 7.0 1.0 2.0 ";

  // Leaf size of 1.
  BinarySpaceTree<HRectBound<2> > rootNode(dataset, 1);

  BOOST_REQUIRE(rootNode.Count() == 6);
  BOOST_REQUIRE(rootNode.Left()->Count() == 3);
  BOOST_REQUIRE(rootNode.Left()->Left()->Count() == 2);
  BOOST_REQUIRE(rootNode.Left()->Left()->Left()->Count() == 1);
  BOOST_REQUIRE(rootNode.Left()->Left()->Right()->Count() == 1);
  BOOST_REQUIRE(rootNode.Left()->Right()->Count() == 1);
  BOOST_REQUIRE(rootNode.Right()->Count() == 3);
  BOOST_REQUIRE(rootNode.Right()->Left()->Count() == 2);
  BOOST_REQUIRE(rootNode.Right()->Left()->Left()->Count() == 1);
  BOOST_REQUIRE(rootNode.Right()->Left()->Right()->Count() == 1);
  BOOST_REQUIRE(rootNode.Right()->Right()->Count() == 1);
}

BOOST_AUTO_TEST_CASE(CheckParents)
{
  arma::mat dataset = "2.0 5.0 9.0 4.0 8.0 7.0;"
                      "3.0 4.0 6.0 7.0 1.0 2.0 ";

  // Leaf size of 1.
  BinarySpaceTree<HRectBound<2> > rootNode(dataset, 1);

  BOOST_REQUIRE_EQUAL(rootNode.Parent(),
      (BinarySpaceTree<HRectBound<2> >*) NULL);
  BOOST_REQUIRE_EQUAL(&rootNode, rootNode.Left()->Parent());
  BOOST_REQUIRE_EQUAL(&rootNode, rootNode.Right()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Left(), rootNode.Left()->Left()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Left(), rootNode.Left()->Right()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Left()->Left(),
      rootNode.Left()->Left()->Left()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Left()->Left(),
      rootNode.Left()->Left()->Right()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Right(), rootNode.Right()->Left()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Right(), rootNode.Right()->Right()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Right()->Left(),
      rootNode.Right()->Left()->Left()->Parent());
  BOOST_REQUIRE_EQUAL(rootNode.Right()->Left(),
      rootNode.Right()->Left()->Right()->Parent());
}

BOOST_AUTO_TEST_CASE(CheckDataset)
{
  arma::mat dataset = "2.0 5.0 9.0 4.0 8.0 7.0;"
                      "3.0 4.0 6.0 7.0 1.0 2.0 ";

  // Leaf size of 1.
  BinarySpaceTree<HRectBound<2> > rootNode(dataset, 1);

  BOOST_REQUIRE_EQUAL(&rootNode.Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Left()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Right()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Left()->Left()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Left()->Right()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Right()->Left()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Right()->Right()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Left()->Left()->Left()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Left()->Left()->Right()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Right()->Left()->Left()->Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(&rootNode.Right()->Left()->Right()->Dataset(), &dataset);
}

// Ensure FurthestDescendantDistance() works.
BOOST_AUTO_TEST_CASE(FurthestDescendantDistanceTest)
{
  arma::mat dataset = "1; 3"; // One point.
  BinarySpaceTree<HRectBound<2> > rootNode(dataset, 1);

  BOOST_REQUIRE_SMALL(rootNode.FurthestDescendantDistance(), 1e-5);

  dataset = "1 -1; 1 -1"; // Square of size [2, 2].

  // Both points are contained in the one node.
  BinarySpaceTree<HRectBound<2> > twoPoint(dataset);
  BOOST_REQUIRE_CLOSE(twoPoint.FurthestDescendantDistance(), sqrt(2.0), 1e-5);
}

// Forward declaration of methods we need for the next test.
template<typename TreeType, typename MatType>
bool CheckPointBounds(TreeType* node, const MatType& data);

template<typename TreeType>
void GenerateVectorOfTree(TreeType* node,
                          size_t depth,
                          std::vector<TreeType*>& v);

template<int t_pow>
bool DoBoundsIntersect(HRectBound<t_pow>& a,
                       HRectBound<t_pow>& b,
                       size_t ia,
                       size_t ib);

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
BOOST_AUTO_TEST_CASE(KdTreeTest)
{
  typedef BinarySpaceTree<HRectBound<2> > TreeType;

  size_t maxRuns = 10; // Ten total tests.
  size_t pointIncrements = 1000; // Range is from 2000 points to 11000.

  // We use the default leaf size of 20.
  for(size_t run = 0; run < maxRuns; run++)
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
    datacopy = dataset; // Save a copy.

    // Build the tree itself.
    TreeType root(dataset, newToOld, oldToNew);

    // Ensure the size of the tree is correct.
    BOOST_REQUIRE_EQUAL(root.Count(), size);

    // Check the forward and backward mappings for correctness.
    for(size_t i = 0; i < size; i++)
    {
      for(size_t j = 0; j < dimensions; j++)
      {
        BOOST_REQUIRE_EQUAL(dataset(j, i), datacopy(j, newToOld[i]));
        BOOST_REQUIRE_EQUAL(dataset(j, oldToNew[i]), datacopy(j, i));
      }
    }

    // Now check that each point is contained inside of all bounds above it.
    CheckPointBounds(&root, dataset);

    // Now check that no peers overlap.
    std::vector<TreeType*> v;
    GenerateVectorOfTree(&root, 1, v);

    // Start with the first pair.
    size_t depth = 2;
    // Compare each peer against every other peer.
    while (depth < v.size())
    {
      for (size_t i = depth; i < 2 * depth && i < v.size(); i++)
        for (size_t j = i + 1; j < 2 * depth && j < v.size(); j++)
          if (v[i] != NULL && v[j] != NULL)
            BOOST_REQUIRE(!DoBoundsIntersect(v[i]->Bound(), v[j]->Bound(),
                  i, j));

      depth *= 2;
    }
  }

  arma::mat dataset = arma::mat(25, 1000);
  for (size_t col = 0; col < dataset.n_cols; ++col)
    for (size_t row = 0; row < dataset.n_rows; ++row)
      dataset(row, col) = row + col;

  TreeType root(dataset);
  // Check the tree size.
  BOOST_REQUIRE_EQUAL(root.TreeSize(), 127);
  // Check the tree depth.
  BOOST_REQUIRE_EQUAL(root.TreeDepth(), 7);
}

// Recursively checks that each node contains all points that it claims to have.
template<typename TreeType, typename MatType>
bool CheckPointBounds(TreeType* node, const MatType& data)
{
  if (node == NULL) // We have passed a leaf node.
    return true;

  TreeType* left = node->Left();
  TreeType* right = node->Right();

  // Check that each point which this tree claims is actually inside the tree.
  for (size_t index = 0; index < node->NumDescendants(); index++)
    if (!node->Bound().Contains(data.col(node->Descendant(index))))
      return false;

  return CheckPointBounds(left, data) && CheckPointBounds(right, data);
}

template<int t_pow>
bool DoBoundsIntersect(HRectBound<t_pow>& a,
                       HRectBound<t_pow>& b,
                       size_t /* ia */,
                       size_t /* ib */)
{
  size_t dimensionality = a.Dim();

  Range r_a;
  Range r_b;

  for (size_t i = 0; i < dimensionality; i++)
  {
    r_a = a[i];
    r_b = b[i];
    if (r_a < r_b || r_a > r_b) // If a does not overlap b at all.
      return false;
  }

  return true;
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

#ifdef ARMA_HAS_SPMAT
// Only run sparse tree tests if we are using Armadillo 3.6.  Armadillo 3.4 has
// some bugs that cause the kd-tree splitting procedure to fail terribly.  Soon,
// that version will be obsolete, though.
#if !((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR == 4))

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
BOOST_AUTO_TEST_CASE(ExhaustiveSparseKDTreeTest)
{
  typedef BinarySpaceTree<HRectBound<2>, EmptyStatistic, arma::SpMat<double> >
      TreeType;

  size_t maxRuns = 2; // Two total tests.
  size_t pointIncrements = 200; // Range is from 200 points to 400.

  // We use the default leaf size of 20.
  for(size_t run = 0; run < maxRuns; run++)
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

    // Ensure the size of the tree is correct.
    BOOST_REQUIRE_EQUAL(root.Count(), size);

    // Check the forward and backward mappings for correctness.
    for(size_t i = 0; i < size; i++)
    {
      for(size_t j = 0; j < dimensions; j++)
      {
        BOOST_REQUIRE_EQUAL(dataset(j, i), datacopy(j, newToOld[i]));
        BOOST_REQUIRE_EQUAL(dataset(j, oldToNew[i]), datacopy(j, i));
      }
    }

    // Now check that each point is contained inside of all bounds above it.
    CheckPointBounds(&root, dataset);

    // Now check that no peers overlap.
    std::vector<TreeType*> v;
    GenerateVectorOfTree(&root, 1, v);

    // Start with the first pair.
    size_t depth = 2;
    // Compare each peer against every other peer.
    while (depth < v.size())
    {
      for (size_t i = depth; i < 2 * depth && i < v.size(); i++)
        for (size_t j = i + 1; j < 2 * depth && j < v.size(); j++)
          if (v[i] != NULL && v[j] != NULL)
            BOOST_REQUIRE(!DoBoundsIntersect(v[i]->Bound(), v[j]->Bound(),
                  i, j));

      depth *= 2;
    }
  }

  arma::SpMat<double> dataset(25, 1000);
  for (size_t col = 0; col < dataset.n_cols; ++col)
    for (size_t row = 0; row < dataset.n_rows; ++row)
      dataset(row, col) = row + col;

  TreeType root(dataset);
  // Check the tree size.
  BOOST_REQUIRE_EQUAL(root.TreeSize(), 127);
  // Check the tree depth.
  BOOST_REQUIRE_EQUAL(root.TreeDepth(), 7);
}

#endif // Using Armadillo 3.4.
#endif // ARMA_HAS_SPMAT

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
  BOOST_REQUIRE_EQUAL(found, true);
}

template<typename TreeType, typename MetricType>
void CheckCovering(const TreeType& node)
{
  // Return if a leaf.  No checking necessary.
  if (node.NumChildren() == 0)
    return;

  const arma::mat& dataset = node.Dataset();
  const size_t nodePoint = node.Point();

  // To ensure that this node satisfies the covering principle, we must ensure
  // that the distance to each child is less than pow(base, scale).
  double maxDistance = pow(node.Base(), node.Scale());
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    const size_t childPoint = node.Child(i).Point();

    double distance = MetricType::Evaluate(dataset.col(nodePoint),
        dataset.col(childPoint));

    BOOST_REQUIRE_LE(distance, maxDistance);

    // Check the child.
    CheckCovering<TreeType, MetricType>(node.Child(i));
  }
}

template<typename TreeType, typename MetricType>
void CheckIndividualSeparation(const TreeType& constantNode,
                               const TreeType& node)
{
  // Don't check points at a lower scale.
  if (node.Scale() < constantNode.Scale())
    return;

  // If at a higher scale, recurse.
  if (node.Scale() > constantNode.Scale())
  {
    for (size_t i = 0; i < node.NumChildren(); ++i)
    {
      // Don't recurse into leaves.
      if (node.Child(i).NumChildren() > 0)
        CheckIndividualSeparation<TreeType, MetricType>(constantNode,
            node.Child(i));
    }

    return;
  }

  // Don't compare the same point against itself.
  if (node.Point() == constantNode.Point())
    return;

  // Now we know we are at the same scale, so make the comparison.
  const arma::mat& dataset = constantNode.Dataset();
  const size_t constantPoint = constantNode.Point();
  const size_t nodePoint = node.Point();

  // Make sure the distance is at least the following value (in accordance with
  // the separation principle of cover trees).
  double minDistance = pow(constantNode.Base(),
      constantNode.Scale());

  double distance = MetricType::Evaluate(dataset.col(constantPoint),
      dataset.col(nodePoint));

  BOOST_REQUIRE_GE(distance, minDistance);
}

template<typename TreeType, typename MetricType>
void CheckSeparation(const TreeType& node, const TreeType& root)
{
  // Check the separation between this point and all other points on this scale.
  CheckIndividualSeparation<TreeType, MetricType>(node, root);

  // Check the children, but only if they are not leaves.  Leaves don't need to
  // be checked.
  for (size_t i = 0; i < node.NumChildren(); ++i)
    if (node.Child(i).NumChildren() > 0)
      CheckSeparation<TreeType, MetricType>(node.Child(i), root);
}


/**
 * Create a simple cover tree and then make sure it is valid.
 */
BOOST_AUTO_TEST_CASE(SimpleCoverTreeConstructionTest)
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
  CoverTree<> tree(data); // Expansion constant of 2.0.

  // The furthest point from the root will be (-5, -5), with a distance of
  // of sqrt(50).  This means the scale of the root node should be 3 (because
  // 2^3 = 8).
  BOOST_REQUIRE_EQUAL(tree.Scale(), 3);

  // Now loop through the tree and ensure that each leaf is only created once.
  arma::vec counts;
  counts.zeros(20);
  RecurseTreeCountLeaves(tree, counts);

  // Each point should only have one leaf node representing it.
  for (size_t i = 0; i < 20; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);

  // Each non-leaf should have a self-child.
  CheckSelfChild<CoverTree<> >(tree);

  // Each node must satisfy the covering principle (its children must be less
  // than or equal to a certain distance apart).
  CheckCovering<CoverTree<>, LMetric<2, true> >(tree);

  // Each node's children must be separated by at least a certain value.
  CheckSeparation<CoverTree<>, LMetric<2, true> >(tree, tree);
}

/**
 * Create a large cover tree and make sure it's accurate.
 */
BOOST_AUTO_TEST_CASE(CoverTreeConstructionTest)
{
  arma::mat dataset;
  // 50-dimensional, 1000 point.
  dataset.randu(50, 1000);

  CoverTree<> tree(dataset);

  // Ensure each leaf is only created once.
  arma::vec counts;
  counts.zeros(1000);
  RecurseTreeCountLeaves(tree, counts);

  for (size_t i = 0; i < 1000; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);

  // Each non-leaf should have a self-child.
  CheckSelfChild<CoverTree<> >(tree);

  // Each node must satisfy the covering principle (its children must be less
  // than or equal to a certain distance apart).
  CheckCovering<CoverTree<>, LMetric<2, true> >(tree);

  // Each node's children must be separated by at least a certain value.
  CheckSeparation<CoverTree<>, LMetric<2, true> >(tree, tree);
}

/**
 * Test the manual constructor.
 */
BOOST_AUTO_TEST_CASE(CoverTreeManualConstructorTest)
{
  arma::mat dataset;
  dataset.zeros(10, 10);

  CoverTree<> node(dataset, 1.3, 3, 2, NULL, 1.5, 2.75);

  BOOST_REQUIRE_EQUAL(&node.Dataset(), &dataset);
  BOOST_REQUIRE_EQUAL(node.Base(), 1.3);
  BOOST_REQUIRE_EQUAL(node.Point(), 3);
  BOOST_REQUIRE_EQUAL(node.Scale(), 2);
  BOOST_REQUIRE_EQUAL(node.Parent(), (CoverTree<>*) NULL);
  BOOST_REQUIRE_EQUAL(node.ParentDistance(), 1.5);
  BOOST_REQUIRE_EQUAL(node.FurthestDescendantDistance(), 2.75);
}

/**
 * Make sure cover trees work in different metric spaces.
 */
BOOST_AUTO_TEST_CASE(CoverTreeAlternateMetricTest)
{
  arma::mat dataset;
  // 5-dimensional, 300-point dataset.
  dataset.randu(5, 300);

  CoverTree<LMetric<1, true> > tree(dataset);

  // Ensure each leaf is only created once.
  arma::vec counts;
  counts.zeros(300);
  RecurseTreeCountLeaves<CoverTree<LMetric<1, true> > >(tree, counts);

  for (size_t i = 0; i < 300; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);

  // Each non-leaf should have a self-child.
  CheckSelfChild<CoverTree<LMetric<1, true> > >(tree);

  // Each node must satisfy the covering principle (its children must be less
  // than or equal to a certain distance apart).
  CheckCovering<CoverTree<LMetric<1, true> >, LMetric<1, true> >(tree);

  // Each node's children must be separated by at least a certain value.
  CheckSeparation<CoverTree<LMetric<1, true> >, LMetric<1, true> >(tree, tree);
}

/**
 * Make sure copy constructor works for the cover tree.
 */
BOOST_AUTO_TEST_CASE(CoverTreeCopyConstructor)
{
  arma::mat dataset;
  dataset.randu(10, 10); // dataset is irrelevant.
  CoverTree<> c(dataset, 1.3, 0, 5, NULL, 1.45, 5.2); // Random parameters.
  c.Children().push_back(new CoverTree<>(dataset, 1.3, 1, 4, &c, 1.3, 2.45));
  c.Children().push_back(new CoverTree<>(dataset, 1.5, 2, 3, &c, 1.2, 5.67));

  CoverTree<> d = c;

  // Check that everything is the same.
  BOOST_REQUIRE_EQUAL(c.Dataset().memptr(), d.Dataset().memptr());
  BOOST_REQUIRE_CLOSE(c.Base(), d.Base(), 1e-50);
  BOOST_REQUIRE_EQUAL(c.Point(), d.Point());
  BOOST_REQUIRE_EQUAL(c.Scale(), d.Scale());
  BOOST_REQUIRE_EQUAL(c.Parent(), d.Parent());
  BOOST_REQUIRE_EQUAL(c.ParentDistance(), d.ParentDistance());
  BOOST_REQUIRE_EQUAL(c.FurthestDescendantDistance(),
                      d.FurthestDescendantDistance());
  BOOST_REQUIRE_EQUAL(c.NumChildren(), d.NumChildren());
  BOOST_REQUIRE_NE(&c.Child(0), &d.Child(0));
  BOOST_REQUIRE_NE(&c.Child(1), &d.Child(1));

  BOOST_REQUIRE_EQUAL(c.Child(0).Parent(), &c);
  BOOST_REQUIRE_EQUAL(c.Child(1).Parent(), &c);
  BOOST_REQUIRE_EQUAL(d.Child(0).Parent(), &d);
  BOOST_REQUIRE_EQUAL(d.Child(1).Parent(), &d);

  // Check that the children are okay.
  BOOST_REQUIRE_EQUAL(c.Child(0).Dataset().memptr(),
                      d.Child(0).Dataset().memptr());
  BOOST_REQUIRE_CLOSE(c.Child(0).Base(), d.Child(0).Base(), 1e-50);
  BOOST_REQUIRE_EQUAL(c.Child(0).Point(), d.Child(0).Point());
  BOOST_REQUIRE_EQUAL(c.Child(0).Scale(), d.Child(0).Scale());
  BOOST_REQUIRE_EQUAL(c.Child(0).ParentDistance(), d.Child(0).ParentDistance());
  BOOST_REQUIRE_EQUAL(c.Child(0).FurthestDescendantDistance(),
                      d.Child(0).FurthestDescendantDistance());
  BOOST_REQUIRE_EQUAL(c.Child(0).NumChildren(), d.Child(0).NumChildren());

  BOOST_REQUIRE_EQUAL(c.Child(1).Dataset().memptr(),
                      d.Child(1).Dataset().memptr());
  BOOST_REQUIRE_CLOSE(c.Child(1).Base(), d.Child(1).Base(), 1e-50);
  BOOST_REQUIRE_EQUAL(c.Child(1).Point(), d.Child(1).Point());
  BOOST_REQUIRE_EQUAL(c.Child(1).Scale(), d.Child(1).Scale());
  BOOST_REQUIRE_EQUAL(c.Child(1).ParentDistance(), d.Child(1).ParentDistance());
  BOOST_REQUIRE_EQUAL(c.Child(1).FurthestDescendantDistance(),
                      d.Child(1).FurthestDescendantDistance());
  BOOST_REQUIRE_EQUAL(c.Child(1).NumChildren(), d.Child(1).NumChildren());
}

/**
 * Make sure copy constructor works right for the binary space tree.
 */
BOOST_AUTO_TEST_CASE(BinarySpaceTreeCopyConstructor)
{
  arma::mat data("1");
  BinarySpaceTree<HRectBound<2> > b(data);
  b.Begin() = 10;
  b.Count() = 50;
  b.Left() = new BinarySpaceTree<HRectBound<2> >(data);
  b.Left()->Begin() = 10;
  b.Left()->Count() = 30;
  b.Right() = new BinarySpaceTree<HRectBound<2> >(data);
  b.Right()->Begin() = 40;
  b.Right()->Count() = 20;

  // Copy the tree.
  BinarySpaceTree<HRectBound<2> > c(b);

  // Ensure everything copied correctly.
  BOOST_REQUIRE_EQUAL(b.Begin(), c.Begin());
  BOOST_REQUIRE_EQUAL(b.Count(), c.Count());
  BOOST_REQUIRE_NE(b.Left(), c.Left());
  BOOST_REQUIRE_NE(b.Right(), c.Right());

  // Check the children.
  BOOST_REQUIRE_EQUAL(b.Left()->Begin(), c.Left()->Begin());
  BOOST_REQUIRE_EQUAL(b.Left()->Count(), c.Left()->Count());
  BOOST_REQUIRE_EQUAL(b.Left()->Left(),
      (BinarySpaceTree<HRectBound<2> >*) NULL);
  BOOST_REQUIRE_EQUAL(b.Left()->Left(), c.Left()->Left());
  BOOST_REQUIRE_EQUAL(b.Left()->Right(),
      (BinarySpaceTree<HRectBound<2> >*) NULL);
  BOOST_REQUIRE_EQUAL(b.Left()->Right(), c.Left()->Right());

  BOOST_REQUIRE_EQUAL(b.Right()->Begin(), c.Right()->Begin());
  BOOST_REQUIRE_EQUAL(b.Right()->Count(), c.Right()->Count());
  BOOST_REQUIRE_EQUAL(b.Right()->Left(),
      (BinarySpaceTree<HRectBound<2> >*) NULL);
  BOOST_REQUIRE_EQUAL(b.Right()->Left(), c.Right()->Left());
  BOOST_REQUIRE_EQUAL(b.Right()->Right(),
      (BinarySpaceTree<HRectBound<2> >*) NULL);
  BOOST_REQUIRE_EQUAL(b.Right()->Right(), c.Right()->Right());
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
  BOOST_REQUIRE_EQUAL(numLeaves, node->NumDescendants());

  // Now check that each descendant is somewhere in the tree.
  for (size_t i = 0; i < node->NumDescendants(); ++i)
  {
    Log::Debug << "Check for descendant " << node->Descendant(i) << " (i " <<
        i << ").\n";
    BOOST_REQUIRE_EQUAL(FindIndex(node, node->Descendant(i)), true);
  }

  // Now check that every actual descendant is accessible through the
  // Descendant() function.
  BOOST_REQUIRE_EQUAL(CheckAccessibility(node, node), true);

  // Now check that there are no duplicates in the list of descendants.
  std::vector<size_t> descendants;
  descendants.resize(node->NumDescendants());
  for (size_t i = 0; i < node->NumDescendants(); ++i)
    descendants[i] = node->Descendant(i);

  // Sort the list.
  std::sort(descendants.begin(), descendants.end());

  // Check that there are no duplicates (this is easy because it's sorted).
  for (size_t i = 1; i < descendants.size(); ++i)
    BOOST_REQUIRE_NE(descendants[i], descendants[i - 1]);

  // Now perform these same checks for the children.
  for (size_t i = 0; i < node->NumChildren(); ++i)
    CheckDescendants(&node->Child(i));
}

/**
 * Make sure Descendant() and NumDescendants() works properly for the cover
 * tree.
 */
BOOST_AUTO_TEST_CASE(CoverTreeDescendantTest)
{
  arma::mat dataset;
  dataset.randu(3, 100);

  CoverTree<> tree(dataset);

  // Now check that the NumDescendants() count and each Descendant() is right
  // using the recursive function above.
  CheckDescendants(&tree);
}

/*
 * Make sure that constructor for cosine tree is working.
 */
BOOST_AUTO_TEST_CASE(CosineTreeConstructorTest)
{
  // Create test data.
  arma::mat data = arma::randu<arma::mat>(5, 5);
  arma::rowvec centroid = arma::randu<arma::rowvec>(1, 5);
  arma::vec probabilities = arma::randu<arma::vec>(5, 1);

  // Creating a cosine tree.
  CosineTree ct(data, centroid, probabilities);

  const arma::mat& dataRet = ct.Data();
  const arma::rowvec& centroidRet = ct.Centroid();
  const arma::vec& probabilitiesRet = ct.Probabilities();

  // Check correctness of dimensionality of data matrix.
  BOOST_REQUIRE_EQUAL(data.n_cols, dataRet.n_rows);
  BOOST_REQUIRE_EQUAL(data.n_rows, dataRet.n_cols);

  // Check the data matrix.
  for (size_t i = 0; i < data.n_cols; i++)
    for (size_t j = 0; j < data.n_rows; j++)
      BOOST_REQUIRE_CLOSE((double) dataRet(j, i), (double) data(i, j), 1e-5);

  // Check correctness of dimensionality of centroid.
  BOOST_REQUIRE_EQUAL(centroid.n_cols, centroidRet.n_cols);
  BOOST_REQUIRE_EQUAL(centroid.n_rows, centroidRet.n_rows);

  // Check centroid.
  for (size_t i = 0; i < centroid.n_cols; i++)
    BOOST_REQUIRE_CLOSE((double) centroidRet(0, i), (double) centroid(0,i),
        1e-5);

  // Check correctness of dimentionality of sampling probabilities.
  BOOST_REQUIRE_EQUAL(probabilities.n_cols, probabilitiesRet.n_cols);
  BOOST_REQUIRE_EQUAL(probabilities.n_rows, probabilitiesRet.n_rows);

  // Check sampling probabilities.
  for (size_t i = 0; i < probabilities.n_rows; i++)
    BOOST_REQUIRE_CLOSE((double) probabilitiesRet(i, 0), (double)
        probabilities(i, 0), 1e-5);

  // Check pointers of children nodes.
  BOOST_REQUIRE(ct.Right() == NULL);
  BOOST_REQUIRE(ct.Left() == NULL);
}

/**
 * Make sure that CTNode function in Cosine tree builder is working.
 */
BOOST_AUTO_TEST_CASE(CosineTreeEmptyConstructorTest)
{
  // Create a tree through the empty constructor.
  CosineTree ct;

  // Check to make sure it has no children.
  BOOST_REQUIRE(ct.Right() == NULL);
  BOOST_REQUIRE(ct.Left() == NULL);
}

/**
 * Make sure that CTNode function in CosineTreeBuilder is working.
 * This test just validates the dimentionality and data.
 */
BOOST_AUTO_TEST_CASE(CosineTreeBuilderCTNodeTest)
{
  // Create dummy test data.
  arma::mat data = arma::randu<arma::mat>(5, 5);

  // Create a cosine tree builder object.
  CosineTreeBuilder builder;

  // Create a cosine tree object.
  CosineTree ct;

  // Use the builder to create the tree.
  builder.CTNode(data, ct);

  const arma::mat& dataRet = ct.Data();
  const arma::rowvec& centroidRet = ct.Centroid();
  const arma::vec& probabilitiesRet = ct.Probabilities();

  // Check correctness of dimentionality of data.
  BOOST_REQUIRE_EQUAL(data.n_cols, dataRet.n_cols);
  BOOST_REQUIRE_EQUAL(data.n_rows, dataRet.n_rows);

  // Check data.
  for (size_t i = 0; i < data.n_cols; i++)
    for (size_t j = 0; j < data.n_rows; j++)
      BOOST_REQUIRE_CLOSE((double) dataRet(j, i), (double) data(i, j), 1e-5);

  // Check correctness of dimensionality of centroid.
  BOOST_REQUIRE_EQUAL(data.n_rows, centroidRet.n_cols);
  BOOST_REQUIRE_EQUAL(1, centroidRet.n_rows);

  // Check correctness of dimensionality of sampling probabilities.
  BOOST_REQUIRE_EQUAL(1, probabilitiesRet.n_cols);
  BOOST_REQUIRE_EQUAL(data.n_rows, probabilitiesRet.n_rows);

  // Check pointers of children nodes.
  BOOST_REQUIRE(ct.Right() == NULL);
  BOOST_REQUIRE(ct.Left() == NULL);

}

/**
 * Make sure that the centroid is calculated correctly when the cosine tree is
 * built.
 */
BOOST_AUTO_TEST_CASE(CosineTreeBuilderCentroidTest)
{
  // Create dummy test data.
  arma::mat data;
  data << 1.0 << 2.0 << 3.0 << arma::endr
       << 4.0 << 2.0 << 3.0 << arma::endr
       << 2.5 << 3.0 << 2.0 << arma::endr;

  // Expected centroid.
  arma::vec c;
  c << 2.0 << 3.0 << 2.5 << arma::endr;

  // Build the cosine tree.
  CosineTreeBuilder builder;
  CosineTree ct;
  builder.CTNode(data, ct);

  // Get the centroid.
  arma::rowvec centroid = ct.Centroid();

  // Check correctness of the centroid.
  BOOST_REQUIRE_CLOSE((double) c(0, 0), (double) centroid(0, 0), 1e-5);
  BOOST_REQUIRE_CLOSE((double) c(1, 0), (double) centroid(0, 1), 1e-5);
  BOOST_REQUIRE_CLOSE((double) c(2, 0), (double) centroid(0, 2), 1e-5);
}

/**
 * Make sure that the sampling probabilities are calculated correctly when the
 * cosine tree is built.
 */
BOOST_AUTO_TEST_CASE(CosineTreeBuilderProbabilitiesTest)
{
  // Create dummy test data.
  arma::mat data;
  data << 100.0 <<   2.0 <<   3.0 << arma::endr
       << 400.0 <<   2.0 <<   3.0 << arma::endr
       << 200.5 <<   3.0 <<   2.0 << arma::endr;

  // Expected sample probability.
  arma::vec p;
  p << 0.999907 << 0.00899223 << 0.0102295 << arma::endr;

  // Create the cosine tree.
  CosineTreeBuilder builder;
  CosineTree ct;
  builder.CTNode(data, ct);

  // Get the probabilities.
  const arma::vec& probabilities = ct.Probabilities();

  // Check correctness of sampling probabilities.
  BOOST_REQUIRE_CLOSE((double) p(0, 0), (double) probabilities(0, 0), 1e-4);
  BOOST_REQUIRE_CLOSE((double) p(1, 0), (double) probabilities(1, 0), 1e-4);
  BOOST_REQUIRE_CLOSE((double) p(2, 0), (double) probabilities(2, 0), 1e-4);
}

/**
 * Make sure that the cosine tree builder is splitting nodes.
 */
BOOST_AUTO_TEST_CASE(CosineTreeBuilderCTNodeSplitTest)
{
  // Create dummy test data.
  arma::mat data;
  data << 100.0 <<   2.0 <<   3.0 << arma::endr
       << 400.0 <<   2.0 <<   3.0 << arma::endr
       << 200.5 <<   3.0 <<   2.0 << arma::endr;

  // Build a cosine tree root node, and then split it.
  CosineTreeBuilder builder;
  CosineTree root, left, right;
  builder.CTNode(data, root);
  builder.CTNodeSplit(root, left, right);

  // Ensure that there is no data loss.
  BOOST_REQUIRE_EQUAL((left.NumPoints() + right.NumPoints()), root.NumPoints());

  // Ensure that the dimensionality is correct.
  BOOST_REQUIRE_EQUAL(left.Data().n_cols, data.n_cols);
  BOOST_REQUIRE_EQUAL(right.Data().n_cols, data.n_cols);
}

BOOST_AUTO_TEST_SUITE_END();
