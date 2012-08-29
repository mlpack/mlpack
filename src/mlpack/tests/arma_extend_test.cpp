/**
 * @file arma_extend_test.cpp
 * @author Ryan Curtin
 *
 * Test of the MLPACK extensions to Armadillo.
 */

#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(ArmaExtendTest);

/**
 * Make sure we can reshape a matrix in-place without changing anything.
 */
BOOST_AUTO_TEST_CASE(InplaceReshapeColumnTest)
{
  arma::mat X;
  X.randu(1, 10);
  arma::mat oldX = X;

  arma::inplace_reshape(X, 2, 5);

  BOOST_REQUIRE_EQUAL(X.n_rows, 2);
  BOOST_REQUIRE_EQUAL(X.n_cols, 5);
  for (size_t i = 0; i < 10; ++i)
    BOOST_REQUIRE_CLOSE(X[i], oldX[i], 1e-5); // Order should be preserved.
}

/**
 * Make sure we can reshape a large matrix.
 */
BOOST_AUTO_TEST_CASE(InplaceReshapeMatrixTest)
{
  arma::mat X;
  X.randu(8, 10);
  arma::mat oldX = X;

  arma::inplace_reshape(X, 10, 8);

  BOOST_REQUIRE_EQUAL(X.n_rows, 10);
  BOOST_REQUIRE_EQUAL(X.n_cols, 8);
  for (size_t i = 0; i < 80; ++i)
    BOOST_REQUIRE_CLOSE(X[i], oldX[i], 1e-5); // Order should be preserved.
}

BOOST_AUTO_TEST_SUITE_END();
