/**
 * @file tests/dbscan_test.cpp
 * @author Ryan Curtin
 *
 * Test the DBSCAN implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/dbscan/random_point_selection.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::dbscan;
using namespace mlpack::distribution;

TEST_CASE("OneClusterTest", "[DBSCANTest]")
{
  // Make sure that if we have points in the unit box, and if we set epsilon
  // large enough, all points end up as in one cluster.
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<> d(2.0, 2);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters == 1);
  REQUIRE(assignments.n_elem == points.n_cols);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    REQUIRE(assignments[i] == 0);
}

/**
 * When epsilon is small enough, every point returned should be noise.
 */
TEST_CASE("TinyEpsilonTest", "[DBSCANTest]")
{
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<> d(1e-50, 2);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters == 0);
  REQUIRE(assignments.n_elem == points.n_cols);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    REQUIRE(assignments[i] == SIZE_MAX);
}

/**
 * Check that outliers are properly labeled as noise.
 */
TEST_CASE("OutlierTest", "[DBSCANTest]")
{
  arma::mat points(2, 200, arma::fill::randu);

  // Add 3 outliers.
  points.col(15) = arma::vec("10.3 1.6");
  points.col(45) = arma::vec("-100 0.0");
  points.col(101) = arma::vec("1.5 1.5");

  DBSCAN<> d(0.1, 3);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters > 0);
  REQUIRE(assignments.n_elem == points.n_cols);
  REQUIRE(assignments[15] == SIZE_MAX);
  REQUIRE(assignments[45] == SIZE_MAX);
  REQUIRE(assignments[101] == SIZE_MAX);
}

/**
 * Check that the Gaussian clusters are correctly found.
 */
TEST_CASE("GaussiansTest", "[DBSCANTest]")
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

  DBSCAN<> d(2.0, 3);

  arma::Row<size_t> assignments;
  arma::mat centroids;
  const size_t clusters = d.Cluster(points, assignments, centroids);
  REQUIRE(clusters == 3);

  // Our centroids should be close to one of our Gaussians.
  arma::Row<size_t> matches(3);
  matches.fill(3);
  for (size_t j = 0; j < 3; ++j)
  {
    if (arma::norm(g1.Mean() - centroids.col(j)) < 3.0)
      matches(0) = j;
    else if (arma::norm(g2.Mean() - centroids.col(j)) < 3.0)
      matches(1) = j;
    else if (arma::norm(g3.Mean() - centroids.col(j)) < 3.0)
      matches(2) = j;
  }

  REQUIRE(matches(0) != matches(1));
  REQUIRE(matches(1) != matches(2));
  REQUIRE(matches(2) != matches(0));

  REQUIRE(matches(0) != 3);
  REQUIRE(matches(1) != 3);
  REQUIRE(matches(2) != 3);

  for (size_t i = 0; i < 100; ++i)
  {
    // Each point should either be noise or in cluster matches(0).
    REQUIRE(assignments(i) != matches(1));
    REQUIRE(assignments(i) != matches(2));
  }

  for (size_t i = 100; i < 200; ++i)
  {
    REQUIRE(assignments(i) != matches(0));
    REQUIRE(assignments(i) != matches(2));
  }

  for (size_t i = 200; i < 300; ++i)
  {
    REQUIRE(assignments(i) != matches(0));
    REQUIRE(assignments(i) != matches(1));
  }
}

TEST_CASE("OneClusterSingleModeTest", "[DBSCANTest]")
{
  // Make sure that if we have points in the unit box, and if we set epsilon
  // large enough, all points end up as in one cluster.
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<> d(2.0, 2, false);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters == 1);
  REQUIRE(assignments.n_elem == points.n_cols);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    REQUIRE(assignments[i] == 0);
}

/**
 * When epsilon is small enough, every point returned should be noise.
 */
TEST_CASE("TinyEpsilonSingleModeTest", "[DBSCANTest]")
{
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<> d(1e-50, 2, false);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters == 0);
  REQUIRE(assignments.n_elem == points.n_cols);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    REQUIRE(assignments[i] == SIZE_MAX);
}

/**
 * Check that outliers are properly labeled as noise.
 */
TEST_CASE("OutlierSingleModeTest", "[DBSCANTest]")
{
  arma::mat points(2, 200, arma::fill::randu);

  // Add 3 outliers.
  points.col(15) = arma::vec("10.3 1.6");
  points.col(45) = arma::vec("-100 0.0");
  points.col(101) = arma::vec("1.5 1.5");

  DBSCAN<> d(0.1, 3, false);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters > 0);
  REQUIRE(assignments.n_elem == points.n_cols);
  REQUIRE(assignments[15] == SIZE_MAX);
  REQUIRE(assignments[45] == SIZE_MAX);
  REQUIRE(assignments[101] == SIZE_MAX);
}

/**
 * Check that the Gaussian clusters are correctly found.
 */
TEST_CASE("GaussiansSingleModeTest", "[DBSCANTest]")
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

  DBSCAN<> d(2.0, 3);

  arma::Row<size_t> assignments;
  arma::mat centroids;
  const size_t clusters = d.Cluster(points, assignments, centroids);
  REQUIRE(clusters == 3);

  // Our centroids should be close to one of our Gaussians.
  arma::Row<size_t> matches(3);
  matches.fill(3);
  for (size_t j = 0; j < 3; ++j)
  {
    if (arma::norm(g1.Mean() - centroids.col(j)) < 3.0)
      matches(0) = j;
    else if (arma::norm(g2.Mean() - centroids.col(j)) < 3.0)
      matches(1) = j;
    else if (arma::norm(g3.Mean() - centroids.col(j)) < 3.0)
      matches(2) = j;
  }

  REQUIRE(matches(0) != matches(1));
  REQUIRE(matches(1) != matches(2));
  REQUIRE(matches(2) != matches(0));

  REQUIRE(matches(0) != 3);
  REQUIRE(matches(1) != 3);
  REQUIRE(matches(2) != 3);

  for (size_t i = 0; i < 100; ++i)
  {
    // Each point should either be noise or in cluster matches(0).
    REQUIRE(assignments(i) != matches(1));
    REQUIRE(assignments(i) != matches(2));
  }

  for (size_t i = 100; i < 200; ++i)
  {
    REQUIRE(assignments(i) != matches(0));
    REQUIRE(assignments(i) != matches(2));
  }

  for (size_t i = 200; i < 300; ++i)
  {
    REQUIRE(assignments(i) != matches(0));
    REQUIRE(assignments(i) != matches(1));
  }
}

/**
 * Check that OrderedPointSelection works correctly.
 */
TEST_CASE("OrderedPointSelectionTest", "[DBSCANTest]")
{
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<> d(2.0, 2);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters == 1);

  // The number of assignments returned should be the same as points.
  REQUIRE(assignments.n_elem == points.n_cols);
}

/**
 * Check that RandomPointSelection works correctly.
 */
TEST_CASE("RandomPointSelectionTest", "[DBSCANTest]")
{
  arma::mat points(10, 200, arma::fill::randu);

  DBSCAN<RangeSearch<>, RandomPointSelection> d(2.0, 2);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  REQUIRE(clusters == 1);

  // The number of assignments returned should be the same as points.
  REQUIRE(assignments.n_elem == points.n_cols);
}
