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

BOOST_AUTO_TEST_SUITE(DBSCANTest);

BOOST_AUTO_TEST_CASE(OneClusterTest)
{
  // Make sure that if we have points in the unit box, and if we set epsilon
  // large enough, all points end up as in one cluster.
  arma::mat points(10, 1000, arma::fill::randu);

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
  arma::mat points(10, 1000, arma::fill::randu);

  DBSCAN<> d(1e-50, 2);

  arma::Row<size_t> assignments;
  const size_t clusters = d.Cluster(points, assignments);

  BOOST_REQUIRE_EQUAL(clusters, 0);
  BOOST_REQUIRE_EQUAL(assignments.n_elem, points.n_cols);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], SIZE_MAX);
}

BOOST_AUTO_TEST_SUITE_END();
