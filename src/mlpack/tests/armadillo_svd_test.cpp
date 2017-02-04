#include <mlpack/core.hpp>
#include <mlpack/methods/cf/svd_wrapper.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

BOOST_AUTO_TEST_SUITE(ArmadilloSVDTest);

using namespace std;
using namespace mlpack;
using namespace mlpack::cf;
using namespace arma;

/**
 * Test armadillo SVD for normal factorization
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
BOOST_AUTO_TEST_CASE(ArmadilloSVDNormalFactorizationTest)
{
  mat test = randu<mat>(20, 20);

  SVDWrapper<> svd;
  arma::mat W, H, sigma;
  double result = svd.Apply(test, W, sigma, H);

  BOOST_REQUIRE_LT(result, 0.01);

  test = randu<mat>(50, 50);
  result = svd.Apply(test, W, sigma, H);

  BOOST_REQUIRE_LT(result, 0.01);
}

/**
 * Test armadillo SVD for low rank matrix factorization
 */
BOOST_AUTO_TEST_CASE(ArmadilloSVDLowRankFactorizationTest)
{
  mat W_t = randu<mat>(30, 3);
  mat H_t = randu<mat>(3, 40);

  // create a row-rank matrix
  mat test = W_t * H_t;

  SVDWrapper<> svd;
  arma::mat W, H;
  double result = svd.Apply(test, 3, W, H);

  BOOST_REQUIRE_LT(result, 0.01);
}


BOOST_AUTO_TEST_SUITE_END();
