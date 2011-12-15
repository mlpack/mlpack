/**
 * @file tree_test.cpp
 *
 * Tests for tree-building methods.
 */
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <vector>

#include <boost/test/unit_test.hpp>

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

  BOOST_REQUIRE_EQUAL(b.Dim(), 0);
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

  // This will be the Euclidean squared distance.
  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 95.0, 1e-5);

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

  BOOST_REQUIRE_CLOSE(b.MinDistance(c), 22.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.MinDistance(b), 22.0, 1e-5);

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

  // This will be the Euclidean squared distance.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), 253.0, 1e-5);

  point = "2.0 5.0 2.0 -5.0 1.0";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), 46.0, 1e-5);

  point = "1.0 2.0 0.0 -2.0 1.5";

  BOOST_REQUIRE_CLOSE(b.MaxDistance(point), 23.25, 1e-5);
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

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 210.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), 210.0, 1e-5);

  // The other bound is on the edge of the bound.
  c[0] = Range(-2.0, 0.0);
  c[1] = Range(0.0, 1.0);
  c[2] = Range(-3.0, -2.0);
  c[3] = Range(-10.0, -5.0);
  c[4] = Range(2.0, 3.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 134.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), 134.0, 1e-5);

  // The other bound partially overlaps the bound.
  c[0] = Range(-2.0, 1.0);
  c[1] = Range(0.0, 2.0);
  c[2] = Range(-2.0, 2.0);
  c[3] = Range(-8.0, -4.0);
  c[4] = Range(0.0, 4.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 102.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), 102.0, 1e-5);

  // The other bound fully overlaps the bound.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(b), 46.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(c), 61.0, 1e-5);

  // The other bound is entirely inside the bound / the other bound entirely
  // envelops the bound.
  c[0] = Range(-1.0, 3.0);
  c[1] = Range(0.0, 6.0);
  c[2] = Range(-3.0, 3.0);
  c[3] = Range(-7.0, 0.0);
  c[4] = Range(0.0, 5.0);

  BOOST_REQUIRE_CLOSE(b.MaxDistance(c), 100.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(b), 100.0, 1e-5);

  // Identical bounds.  This will be the sum of the squared widths in each
  // dimension.
  BOOST_REQUIRE_CLOSE(b.MaxDistance(b), 46.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.MaxDistance(c), 162.0, 1e-5);

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
    size_t dim = rand() % 20;

    HRectBound<2> a(dim);
    HRectBound<2> b(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec lo_a(dim);
    arma::vec width_a(dim);

    lo_a.randu();
    width_a.randu();

    arma::vec lo_b(dim);
    arma::vec width_b(dim);

    lo_b.randu();
    width_b.randu();

    for (size_t j = 0; j < dim; j++)
    {
      a[j] = Range(lo_a[j], lo_a[j] + width_a[j]);
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
    size_t dim = rand() % 20;

    HRectBound<2> a(dim);

    // We will set the low randomly and the width randomly for each dimension of
    // each bound.
    arma::vec lo_a(dim);
    arma::vec width_a(dim);

    lo_a.randu();
    width_a.randu();

    for (size_t j = 0; j < dim; j++)
      a[j] = Range(lo_a[j], lo_a[j] + width_a[j]);

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
  DBallBound<> b1;
  DBallBound<> b2;

  // Create two balls with a center distance of 1 from each other.
  // Give the first one a radius of 0.3 and the second a radius of 0.4.
  b1.center().set_size(3);
  b1.center()[0] = 1;
  b1.center()[1] = 2;
  b1.center()[2] = 3;
  b1.set_radius(0.3);

  b2.center().set_size(3);
  b2.center()[0] = 1;
  b2.center()[1] = 2;
  b2.center()[2] = 4;
  b2.set_radius(0.4);

  BOOST_REQUIRE_CLOSE(sqrt(b1.MinDistanceSq(b2)), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b1.RangeDistanceSq(b2).Hi()), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b1.RangeDistanceSq(b2).Lo()), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b1.RangeDistance(b2).Hi(), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(b1.RangeDistance(b2).Lo(), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b1.MinToMidSq(b2)), 1-0.3, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b1.MinimaxDistanceSq(b2)), 1-0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b1.MidDistanceSq(b2)), 1.0, 1e-5);

  BOOST_REQUIRE_CLOSE(sqrt(b2.MinDistanceSq(b1)), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.MaxDistanceSq(b1)), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.RangeDistanceSq(b1).Hi()), 1+0.3+0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.RangeDistanceSq(b1).Lo()), 1-0.3-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.MinToMidSq(b1)), 1-0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.MinimaxDistanceSq(b1)), 1-0.4+0.3, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.MidDistanceSq(b1)), 1.0, 1e-5);

  BOOST_REQUIRE(b1.Contains(b1.center()));
  BOOST_REQUIRE(!b1.Contains(b2.center()));

  BOOST_REQUIRE(!b2.Contains(b1.center()));
  BOOST_REQUIRE(b2.Contains(b2.center()));
  arma::vec b2point(3); // A point that's within the radius but not the center.
  b2point[0] = 1.1;
  b2point[1] = 2.1;
  b2point[2] = 4.1;

  BOOST_REQUIRE(b2.Contains(b2point));

  BOOST_REQUIRE_SMALL(sqrt(b1.MinDistanceSq(b1.center())), 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b1.MinDistanceSq(b2.center())), 1 - 0.3, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.MinDistanceSq(b1.center())), 1 - 0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b2.MaxDistanceSq(b1.center())), 1 + 0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(sqrt(b1.MaxDistanceSq(b2.center())), 1 + 0.3, 1e-5);
}

/**
 * Ensure that a bound, by default, is empty and has no dimensionality, and the
 * box size vector is empty.
 */
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundEmptyConstructor)
{
  PeriodicHRectBound<2> b;

  BOOST_REQUIRE_EQUAL(b.Dim(), 0);
  BOOST_REQUIRE_EQUAL(b.box().n_elem, 0);
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
  BOOST_REQUIRE_EQUAL(b.box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(b.box()[0], 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b.box()[1], 6.0, 1e-5);

  PeriodicHRectBound<2> d(arma::vec("2 3 4 5 6"));

  BOOST_REQUIRE_EQUAL(d.Dim(), 5);
  BOOST_REQUIRE_SMALL(d[0].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[1].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[2].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[3].Width(), 1e-5);
  BOOST_REQUIRE_SMALL(d[4].Width(), 1e-5);
  BOOST_REQUIRE_EQUAL(d.box().n_elem, 5);
  BOOST_REQUIRE_CLOSE(d.box()[0], 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.box()[1], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.box()[2], 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.box()[3], 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(d.box()[4], 6.0, 1e-5);
}

/**
 * Test the copy constructor.
 *
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
  BOOST_REQUIRE_EQUAL(c.box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(c.box()[0], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.box()[1], 4.0, 1e-5);
}*/

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
  BOOST_REQUIRE_EQUAL(c.box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(c.box()[0], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(c.box()[1], 4.0, 1e-5);
}*/

/**
 * Ensure that we can set the box size correctly.
 *
BOOST_AUTO_TEST_CASE(PeriodicHRectBoundSetBoxSize)
{
  PeriodicHRectBound<2> b(arma::vec("1 2"));

  b.SetBoxSize(arma::vec("10 12"));

  BOOST_REQUIRE_EQUAL(b.box().n_elem, 2);
  BOOST_REQUIRE_CLOSE(b.box()[0], 10.0, 1e-5);
  BOOST_REQUIRE_CLOSE(b.box()[1], 12.0, 1e-5);
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
  b = PeriodicHRectBound<2>(arma::vec("5.0"));
  point.set_size(1);

  b[0] = Range("2.0 4.0"); // Entirely inside box.
  point[0] = 7.5; // Inside first right image of the box.

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);

  b[0] = Range("0.0 5.0"); // Fills box fully.
  point[1] = 19.3; // Inside the box, which covers everything.

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);

  b[0] = Range("-10.0 10.0"); // Larger than the box.
  point[0] = -500.0; // Inside the box, which covers everything.

  BOOST_REQUIRE_SMALL(b.MinDistance(point), 1e-5);

  b[0] = Range("-2.0 1.0"); // Crosses over an edge.
  point[0] = 2.9; // The first right image of the bound starts at 3.0.

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 0.01, 1e-5);

  b[0] = Range("2.0 4.0"); // Inside box.
  point[0] = 0.0; // Closest to the first left image of the bound.

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 1.0, 1e-5);

  b[0] = Range("0.0 2.0"); // On edge of box.
  point[0] = 7.1; // 0.1 away from the first right image of the bound.

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 0.01, 1e-5);

  b[0] = Range("-10.0 10.0"); // Box is of infinite size.
  point[0] = 810.0; // 800 away from the only image of the box.

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 640000, 1e-5);

  b[0] = Range("2.0 4.0"); // Box size of -5 should function the same as 5.
  point[0] = -10.8; // Should alias to 4.2.

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 0.04, 1e-5);

  // Switch our bound to a higher dimensionality.  This should ensure that the
  // dimensions are independent like they should be.
  b = PeriodicHRectBound<2>(arma::vec("5.0 5.0 5.0 5.0 5.0 5.0 0.0 -5.0"));

  b[0] = Range("2.0 4.0"); // Entirely inside box.
  b[1] = Range("0.0 5.0"); // Fills box fully.
  b[2] = Range("-10.0 10.0"); // Larger than the box.
  b[3] = Range("-2.0 1.0"); // Crosses over an edge.
  b[4] = Range("2.0 4.0"); // Inside box.
  b[5] = Range("0.0 2.0"); // On edge of box.
  b[6] = Range("-10.0 10.0"); // Box is of infinite size.
  b[7] = Range("2.0 4.0"); // Box size of -5 should function the same as 5.

  point.set_size(8);
  point[0] = 7.5; // Inside first right image of the box.
  point[1] = 19.3; // Inside the box, which covers everything.
  point[2] = -500.0; // Inside the box, which covers everything.
  point[3] = 2.9; // The first right image of the bound starts at 3.0.
  point[4] = 0.0; // Closest to the first left image of the bound.
  point[5] = 7.1; // 0.1 away from the first right image of the bound.
  point[6] = 810.0; // 800 away from the only image of the box.
  point[7] = -10.8; // Should alias to 4.2.

  BOOST_REQUIRE_CLOSE(b.MinDistance(point), 640001.06, 1e-10);
}*/

/**
 * It seems as though Bill has stumbled across a bug where
 * BinarySpaceTree<>::count() returns something different than
 * BinarySpaceTree<>::count_.  So, let's build a simple tree and make sure they
 * are the same.
 */
BOOST_AUTO_TEST_CASE(tree_count_mismatch)
{
  arma::mat dataset = "2.0 5.0 9.0 4.0 8.0 7.0;"
                      "3.0 4.0 6.0 7.0 1.0 2.0 ";

  // Leaf size of 1.
  BinarySpaceTree<HRectBound<2> > root_node(dataset, 1);

  BOOST_REQUIRE(root_node.Count() == 6);
  BOOST_REQUIRE(root_node.Left()->Count() == 3);
  BOOST_REQUIRE(root_node.Left()->Left()->Count() == 2);
  BOOST_REQUIRE(root_node.Left()->Left()->Left()->Count() == 1);
  BOOST_REQUIRE(root_node.Left()->Left()->Right()->Count() == 1);
  BOOST_REQUIRE(root_node.Left()->Right()->Count() == 1);
  BOOST_REQUIRE(root_node.Right()->Count() == 3);
  BOOST_REQUIRE(root_node.Right()->Left()->Count() == 2);
  BOOST_REQUIRE(root_node.Right()->Left()->Left()->Count() == 1);
  BOOST_REQUIRE(root_node.Right()->Left()->Right()->Count() == 1);
  BOOST_REQUIRE(root_node.Right()->Right()->Count() == 1);
}

// Forward declaration of methods we need for the next test.
template<typename TreeType>
bool CheckPointBounds(TreeType* node, const arma::mat& data);

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
BOOST_AUTO_TEST_CASE(kd_tree_test)
{
  typedef BinarySpaceTree<HRectBound<2> > TreeType;

  size_t max_runs = 10; // Ten total tests.
  size_t point_increments = 1000; // Range is from 2000 points to 11000.

  // We use the default leaf size of 20.
  for(size_t run = 0; run < max_runs; run++)
  {
    size_t dimensions = run + 2;
    size_t max_points = (run + 1) * point_increments;

    size_t size = max_points;
    arma::mat dataset = arma::mat(dimensions, size);
    arma::mat datacopy; // Used to test mappings.

    // Mappings for post-sort verification of data.
    std::vector<size_t> new_to_old;
    std::vector<size_t> old_to_new;

    // Generate data.
    dataset.randu();
    datacopy = dataset; // Save a copy.

    // Build the tree itself.
    TreeType root(dataset, new_to_old, old_to_new);

    // Ensure the size of the tree is correct.
    BOOST_REQUIRE_EQUAL(root.Count(), size);

    // Check the forward and backward mappings for correctness.
    for(size_t i = 0; i < size; i++)
    {
      for(size_t j = 0; j < dimensions; j++)
      {
        BOOST_REQUIRE_EQUAL(dataset(j, i), datacopy(j, new_to_old[i]));
        BOOST_REQUIRE_EQUAL(dataset(j, old_to_new[i]), datacopy(j, i));
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
template<typename TreeType>
bool CheckPointBounds(TreeType* node, const arma::mat& data)
{
  if (node == NULL) // We have passed a leaf node.
    return true;

  TreeType* left = node->Left();
  TreeType* right = node->Right();

  size_t begin = node->Begin();
  size_t count = node->Count();

  // Check that each point which this tree claims is actually inside the tree.
  for (size_t index = begin; index < begin + count; index++)
  {
    if (!node->Bound().Contains(data.col(index)))
      return false;
  }

  return CheckPointBounds(left, data) && CheckPointBounds(right, data);
}

template<int t_pow>
bool DoBoundsIntersect(HRectBound<t_pow>& a,
                       HRectBound<t_pow>& b,
                       size_t ia,
                       size_t ib) {
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

  size_t max_runs = 2; // Two total tests.
  size_t point_increments = 200; // Range is from 200 points to 400.

  // We use the default leaf size of 20.
  for(size_t run = 0; run < max_runs; run++)
  {
    size_t dimensions = run + 2;
    size_t max_points = (run + 1) * point_increments;

    size_t size = max_points;
    arma::SpMat<double> dataset = arma::SpMat<double>(dimensions, size);
    arma::SpMat<double> datacopy; // Used to test mappings.

    // Mappings for post-sort verification of data.
    std::vector<size_t> new_to_old;
    std::vector<size_t> old_to_new;

    // Generate data.
    dataset.randu();
    datacopy = dataset; // Save a copy.

    // Build the tree itself.
    TreeType root(dataset, new_to_old, old_to_new);

    // Ensure the size of the tree is correct.
    BOOST_REQUIRE_EQUAL(root.Count(), size);

    // Check the forward and backward mappings for correctness.
    for(size_t i = 0; i < size; i++)
    {
      for(size_t j = 0; j < dimensions; j++)
      {
        BOOST_REQUIRE_EQUAL(dataset(j, i), datacopy(j, new_to_old[i]));
        BOOST_REQUIRE_EQUAL(dataset(j, old_to_new[i]), datacopy(j, i));
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

BOOST_AUTO_TEST_SUITE_END();
