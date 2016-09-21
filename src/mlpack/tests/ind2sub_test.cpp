/**
 * @file ind2sub_test.cpp
 * @author Nilay Jain
 *
 * Test the backported Armadillo ind2sub() and sub2ind() functions.
 */
#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

BOOST_AUTO_TEST_SUITE(ind2subTest);

/**
 * This test checks whether ind2sub and sub2ind are 
 * compiled successfully and that they function properly.
 */
BOOST_AUTO_TEST_CASE(ind2sub_test)
{
  arma::mat A = arma::randu(4,5);
  size_t index = 13;
  arma::uvec u = arma::ind2sub(arma::size(A), index);

  BOOST_REQUIRE_EQUAL(u(0), index % A.n_rows);
  BOOST_REQUIRE_EQUAL(u(1), index / A.n_rows);

  index = arma::sub2ind(arma::size(A), u(0), u(1));
  BOOST_REQUIRE_EQUAL(index, u(0) + u(1) * A.n_rows);
}

BOOST_AUTO_TEST_SUITE_END();
