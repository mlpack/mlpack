/**
 * @file qdafn_test.cpp
 * @author Ryan Curtin
 *
 * Test the QDAFN functionality.
 */
#define BOOST_TEST_MODULE QDAFNTest

#include <boost/test/unit_test.hpp>

#include <mlpack/core.hpp>
#include "qdafn.hpp"
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace qdafn;
using namespace mlpack::neighbor;

/**
 * With one reference point, make sure that is the one that is returned.
 */
BOOST_AUTO_TEST_CASE(QDAFNTrivialTest)
{
  arma::mat refSet(5, 1);
  refSet.randu();

  // 5 tables, 1 point.
  QDAFN<> qdafn(refSet, 5, 1);

  arma::mat querySet(5, 5);
  querySet.randu();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  qdafn.Search(querySet, 1, neighbors, distances);

  // Check sizes.
  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 1);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 5);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 1);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 5);

  for (size_t i = 0; i < 5; ++i)
  {
    BOOST_REQUIRE_EQUAL(neighbors[i], 0);
    const double dist = metric::EuclideanDistance::Evaluate(querySet.col(i),
        refSet.col(0));
    BOOST_REQUIRE_CLOSE(distances[i], dist, 1e-5);
  }
}

/**
 * Given a random uniform reference set, ensure that we get a neighbor and
 * distance within 10% of the actual true furthest neighbor distance at least
 * 70% of the time.
 */
BOOST_AUTO_TEST_CASE(QDAFNUniformSet)
{
  arma::mat uniformSet = arma::randu<arma::mat>(25, 1000);

  QDAFN<> qdafn(uniformSet, 10, 30);

  // Get the actual neighbors.
  AllkFN kfn(uniformSet);
  arma::Mat<size_t> trueNeighbors;
  arma::mat trueDistances;

  kfn.Search(1000, trueNeighbors, trueDistances);

  arma::Mat<size_t> qdafnNeighbors;
  arma::mat qdafnDistances;

  qdafn.Search(uniformSet, 1, qdafnNeighbors, qdafnDistances);

  BOOST_REQUIRE_EQUAL(qdafnNeighbors.n_rows, 1);
  BOOST_REQUIRE_EQUAL(qdafnNeighbors.n_cols, 1000);
  BOOST_REQUIRE_EQUAL(qdafnDistances.n_rows, 1);
  BOOST_REQUIRE_EQUAL(qdafnDistances.n_cols, 1000);

  size_t successes = 0;
  for (size_t i = 0; i < 1000; ++i)
  {
    // Find the true neighbor.
    size_t trueIndex = 1000;
    for (size_t j = 0; j < 1000; ++j)
    {
      if (trueNeighbors(j, i) == qdafnNeighbors(0, i))
      {
        trueIndex = j;
        break;
      }
    }

    BOOST_REQUIRE_NE(trueIndex, 1000);
    if (0.9 * trueDistances(0, i) <= qdafnDistances(0, i))
      ++successes;
  }

  BOOST_REQUIRE_GE(successes, 700);
}
