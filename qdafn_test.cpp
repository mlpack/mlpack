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

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace qdafn;

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
