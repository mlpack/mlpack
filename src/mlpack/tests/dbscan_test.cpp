/**
 * @file dbscan_test.cpp
 * @author Ryan Curtin
 *
 * Test the DBSCAN implementation.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::dbscan;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(DBSCANTest);

BOOST_AUTO_TEST_CASE(OneClusterTest)
{
  // Make sure that if we have points in the unit box, and if we set epsilon
  // large enough, all points end up as in one cluster.
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<> d(2.0, 2);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  BOOST_REQUIRE_EQUAL(clusters, 1);
  BOOST_REQUIRE_EQUAL(assignments.n_elem, points.n_cols);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 0);
}

/**
 * When epsilon is small enough, every point returned should be noise.
 */
BOOST_AUTO_TEST_CASE(TinyEpsilonTest)
{
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<> d(1e-50, 2);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  BOOST_REQUIRE_EQUAL(clusters, 0);
  BOOST_REQUIRE_EQUAL(assignments.n_elem, points.n_cols);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], SIZE_MAX);
}

/**
 * Check that outliers are properly labeled as noise.
 */
BOOST_AUTO_TEST_CASE(OutlierTest)
{
  arma::mat points(2, 200, arma::fill::randu);

  // Add 3 outliers.
  points.col(15) = arma::vec("10.3 1.6");
  points.col(45) = arma::vec("-100 0.0");
  points.col(101) = arma::vec("1.5 1.5");

  DBSCAN<> d(0.1, 3);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  BOOST_REQUIRE_GT(clusters, 0);
  BOOST_REQUIRE_EQUAL(assignments.n_elem, points.n_cols);
  BOOST_REQUIRE_EQUAL(assignments[15], SIZE_MAX);
  BOOST_REQUIRE_EQUAL(assignments[45], SIZE_MAX);
  BOOST_REQUIRE_EQUAL(assignments[101], SIZE_MAX);
}

/**
 * Check that the Gaussian clusters are correctly found.
 */
BOOST_AUTO_TEST_CASE(GaussiansTest)
{
  arma::mat points(3, 300);

  GaussianDistribution g1(3), g2(3), g3(3);
  g1.Mean() = arma::vec("0.0 0.0 0.0");
  g2.Mean() = arma::vec("6.0 6.0 8.0");
  g3.Mean() = arma::vec("-6.0 1.0 -7.0");
  for (size_t i = 0; i < 100; ++i)
    points.col(i) = g1.Random();
  for (size_t i = 100; i < 200; ++i)
    points.col(i) = g2.Random();
  for (size_t i = 200; i < 300; ++i)
    points.col(i) = g3.Random();

  DBSCAN<> d(1.0, 3);

  arma::Row<size_t> assignments;
  arma::mat centroids;
  const size_t clusters = d.Cluster(points, assignments, centroids);
  BOOST_REQUIRE_EQUAL(clusters, 3);

  // Our centroids should be close to one of our Gaussians.
  arma::Row<size_t> matches(3);
  matches.fill(3);
  for (size_t j = 0; j < 3; ++j)
  {
    if (arma::norm(g1.Mean() - centroids.col(j)) < 1.0)
      matches(j) = 0;
    else if (arma::norm(g2.Mean() - centroids.col(j)) < 1.0)
      matches(j) = 1;
    else if (arma::norm(g3.Mean() - centroids.col(j)) < 1.0)
      matches(j) = 2;

    BOOST_REQUIRE_NE(matches(j), 3);
  }

  BOOST_REQUIRE_NE(matches(0), matches(1));
  BOOST_REQUIRE_NE(matches(1), matches(2));
  BOOST_REQUIRE_NE(matches(2), matches(0));

  for (size_t i = 0; i < 100; ++i)
  {
    // Each point should either be noise or in cluster matches(0).
    BOOST_REQUIRE_NE(assignments(i), matches(1));
    BOOST_REQUIRE_NE(assignments(i), matches(2));
  }

  for (size_t i = 100; i < 200; ++i)
  {
    BOOST_REQUIRE_NE(assignments(i), matches(0));
    BOOST_REQUIRE_NE(assignments(i), matches(2));
  }

  for (size_t i = 200; i < 300; ++i)
  {
    BOOST_REQUIRE_NE(assignments(i), matches(0));
    BOOST_REQUIRE_NE(assignments(i), matches(1));
  }
}

BOOST_AUTO_TEST_SUITE_END();
