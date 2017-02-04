/**
 * @file hyperplane_test.cpp
 *
 * Tests for Hyperplane and ProjVector implementations.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/space_split/hyperplane.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(HyperplaneTest);

/**
 * Ensure that a hyperplane, by default, consider all points to the left.
 */
BOOST_AUTO_TEST_CASE(HyperplaneEmptyConstructor)
{
  Hyperplane<EuclideanDistance> h1;
  AxisOrthogonalHyperplane<EuclideanDistance> h2;

  arma::mat dataset;
  dataset.randu(3, 20); // 20 points in 3 dimensions.

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    BOOST_REQUIRE(h1.Left(dataset.col(i)));
    BOOST_REQUIRE(h2.Left(dataset.col(i)));
    BOOST_REQUIRE(!h1.Right(dataset.col(i)));
    BOOST_REQUIRE(!h2.Right(dataset.col(i)));
  }
}

/**
 * Ensure that we get the correct hyperplane given the projection vector.
 */
BOOST_AUTO_TEST_CASE(ProjectionTest)
{
  // General hyperplane.
  ProjVector projVect1(arma::vec("1 1"));
  Hyperplane<EuclideanDistance> h1(projVect1, 0);

  BOOST_REQUIRE_EQUAL(h1.Project(arma::vec("1 -1")), 0);
  BOOST_REQUIRE(h1.Left(arma::vec("1 -1")));
  BOOST_REQUIRE(!h1.Right(arma::vec("1 -1")));

  BOOST_REQUIRE_EQUAL(h1.Project(arma::vec("-1 1")), 0);
  BOOST_REQUIRE(h1.Left(arma::vec("-1 1")));
  BOOST_REQUIRE(!h1.Right(arma::vec("-1 1")));

  BOOST_REQUIRE_EQUAL(h1.Project(arma::vec("1 0")),
      h1.Project(arma::vec("0 1")));
  BOOST_REQUIRE(h1.Right(arma::vec("1 0")));
  BOOST_REQUIRE(!h1.Left(arma::vec("1 0")));

  BOOST_REQUIRE_EQUAL(h1.Project(arma::vec("-1 -1")),
      h1.Project(arma::vec("-2 0")));
  BOOST_REQUIRE(h1.Left(arma::vec("-1 -1")));
  BOOST_REQUIRE(!h1.Right(arma::vec("-1 -1")));

  // A simple 2-dimensional bound.
  BallBound<EuclideanDistance> b1(2);

  b1.Center() = arma::vec("-1 -1");
  b1.Radius() = 1.41;
  BOOST_REQUIRE(h1.Left(b1));
  BOOST_REQUIRE(!h1.Right(b1));

  b1.Center() = arma::vec("1 1");
  b1.Radius() = 1.41;
  BOOST_REQUIRE(h1.Right(b1));
  BOOST_REQUIRE(!h1.Left(b1));

  b1.Center() = arma::vec("0 0");
  b1.Radius() = 1.41;
  BOOST_REQUIRE(!h1.Right(b1));
  BOOST_REQUIRE(!h1.Left(b1));
}

/**
 * Ensure that we get the correct AxisOrthogonalHyperplane given the
 * AxisParallelProjVector.
 */
BOOST_AUTO_TEST_CASE(AxisOrthogonalProjectionTest)
{
  // AxisParallel hyperplane.
  AxisParallelProjVector projVect2(1);
  AxisOrthogonalHyperplane<EuclideanDistance> h2(projVect2, 1);

  BOOST_REQUIRE_EQUAL(h2.Project(arma::vec("0 0")), -1);
  BOOST_REQUIRE(h2.Left(arma::vec("0 0")));
  BOOST_REQUIRE(!h2.Right(arma::vec("0 0")));

  BOOST_REQUIRE_EQUAL(h2.Project(arma::vec("0 1")), 0);
  BOOST_REQUIRE(h2.Left(arma::vec("0 1")));
  BOOST_REQUIRE(!h2.Right(arma::vec("0 1")));

  BOOST_REQUIRE_EQUAL(h2.Project(arma::vec("0 2")), 1);
  BOOST_REQUIRE(h2.Right(arma::vec("0 2")));
  BOOST_REQUIRE(!h2.Left(arma::vec("0 2")));

  BOOST_REQUIRE_EQUAL(h2.Project(arma::vec("1 2")), 1);
  BOOST_REQUIRE(h2.Right(arma::vec("1 2")));
  BOOST_REQUIRE(!h2.Left(arma::vec("1 2")));

  BOOST_REQUIRE_EQUAL(h2.Project(arma::vec("1 0")), -1);
  BOOST_REQUIRE(h2.Left(arma::vec("1 0")));
  BOOST_REQUIRE(!h2.Right(arma::vec("1 0")));

  // A simple 2-dimensional bound.
  HRectBound<EuclideanDistance> b2(2);

  b2[0] = Range(-1.0, 1.0);
  b2[1] = Range(-1.0, 1.0);
  BOOST_REQUIRE(h2.Left(b2));
  BOOST_REQUIRE(!h2.Right(b2));

  b2[0] = Range(-1.0, 1.0);
  b2[1] = Range(1.001, 2.0);
  BOOST_REQUIRE(h2.Right(b2));
  BOOST_REQUIRE(!h2.Left(b2));

  b2[0] = Range(-1.0, 1.0);
  b2[1] = Range(0, 2.0);
  BOOST_REQUIRE(!h2.Right(b2));
  BOOST_REQUIRE(!h2.Left(b2));
}

BOOST_AUTO_TEST_SUITE_END();
