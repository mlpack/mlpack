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
  mlpack::math::RandomSeed(10);
  mat test = randu<mat>(5,4);

  PlainSVD svd;
  arma::mat W, H, sigma;
  double result = svd.Apply(test, W, sigma, H);
  
  BOOST_REQUIRE_LT(result, 1e-15);
}

/**
 * Test PlainSVD as wrapper for CF.
 */
BOOST_AUTO_TEST_CASE(PlainSVDCFWrapperTest)
{
  mlpack::math::RandomSeed(10);
  mat test = randu<mat>(5,4);
  
  PlainSVD svd;
  mat W, H;
  double result = svd.Apply(test, 3, W, H);
  
  BOOST_REQUIRE_LT(result, 0.1);
}

BOOST_AUTO_TEST_SUITE_END();
