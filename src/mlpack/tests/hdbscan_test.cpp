/**
 * @file hdbscan_test.cpp
 *
 * Test the HDBSCAN implementation.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hdbscan/hdbscan.hpp>


#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::hdbscan;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(HDBSCANTest);

BOOST_AUTO_TEST_CASE(multipleClusterTest)
{
  // Generate 5000 points on 0.1 radius circle.
  arma::mat points(2, 5000);
  double pi = 3.14156;
  double theeta;
  for (size_t i = 0; i < 5000; i++)
  {
    theeta = (double)i/5000.0;
    points(0, i) = 0.001 * (i / 1000 + 1) * sin(2*pi*theeta);
    points(1, i) = 0.001 * (i / 1000 + 1) * cos(2*pi*theeta);
  }

  HDBSCAN<> h1(5, true);
  arma::Row<size_t> assignments;
  h1.Cluster(points, assignments);

  // Some points in the circle will get classified as noise
  for (size_t i = 0; i < 5; i++)
  {
    for (size_t j = 1; j < 1000; j++)
      BOOST_REQUIRE_EQUAL(assignments(i * 1000 + j),
       assignments(i * 1000));
  }
}

BOOST_AUTO_TEST_CASE(singleClusterTest)
{
  // Generate 5000 points on 0.1 radius circle.
  arma::mat points(2, 5000);
  double pi = 3.14156;
  double theeta;
  for (size_t i = 0; i < 5000; i++)
  {
    theeta = (double)i/5000.0;
    points(0, i) = 100 * sin(2*pi*theeta);
    points(1, i) = 100 * cos(2*pi*theeta);
  }

  HDBSCAN<> h1(5, true);
  arma::Row<size_t> assignments;
  h1.Cluster(points, assignments);

  // Some points in the circle will get classified as noise
  for (size_t i = 0; i < 5000; i++)
  {
    BOOST_REQUIRE((assignments(i) == 0) || (assignments(i) == SIZE_MAX));
  }
}

BOOST_AUTO_TEST_CASE(noiseTest)
{
  // Generate 5000 random points on a unit circle.
  arma::mat points(2, 5000);
  double pi = 3.14156;
  double theeta;
  for (size_t i = 0; i < 5000; i++)
  {
    theeta = math::Random();
    points(0, i) = sin(2*pi*theeta);
    points(1, i) = cos(2*pi*theeta);
  }

  points.resize(points.n_rows, points.n_cols+1);
  points(0, points.n_cols-1) = 100;
  points(1, points.n_cols-1) = 100;

  HDBSCAN<> h1(5);
  arma::Row<size_t> assignments;
  h1.Cluster(points, assignments);

  // The last point must be noise,
  BOOST_REQUIRE_EQUAL(assignments(assignments.n_cols-1), SIZE_MAX);
}

/**
 * Check that the Gaussian clusters are correctly found.
 * Inspired from DBSCAN test suite.
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

  HDBSCAN<> h(5, true);
  arma::Row<size_t> assignments;

  h.Cluster(points, assignments);

  for (size_t i = 0; i < assignments.n_cols; i++)
  {
    // Points 0-99 belong to same cluster.
    // Points 100-199 belong to same cluster.
    // Points 200-299 belong to same cluster.
    BOOST_REQUIRE(assignments(int(i / 100) * 100) == assignments(i));
  }
}

BOOST_AUTO_TEST_SUITE_END();
