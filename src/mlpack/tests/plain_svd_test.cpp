#include <mlpack/core.hpp>
#include <mlpack/methods/cf/plain_svd.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(PlainSVDTest);

using namespace std;
using namespace mlpack;
using namespace mlpack::svd;
using namespace arma;

/**
 * Test PlainSVD for normal factorization
 */
BOOST_AUTO_TEST_CASE(PlainSVDNormalFactorizationTest)
{
  mat test = randu<mat>(20, 20);

  PlainSVD svd;
  arma::mat W, H, sigma;
  double result = svd.Apply(test, W, sigma, H);
  
  BOOST_REQUIRE_LT(result, 0.01);
  
  test = randu<mat>(50, 50);
  result = svd.Apply(test, W, sigma, H);
  
  BOOST_REQUIRE_LT(result, 0.01);
}

/**
 * Test PlainSVD for low rank matrix factorization
 */
BOOST_AUTO_TEST_CASE(PlainSVDLowRankFactorizationTest)
{
  mat W_t = randu<mat>(30, 3);
  mat H_t = randu<mat>(3, 40);
  
  mat test = W_t * H_t;

  PlainSVD svd;
  arma::mat W, H;
  double result = svd.Apply(test, 3, W, H);
  
  BOOST_REQUIRE_LT(result, 0.01);
}


BOOST_AUTO_TEST_SUITE_END();
