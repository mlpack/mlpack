/**
 * @file nmf_test.cpp
 * @author Mohan Rajendran
 *
 * Test file for NMF class.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/nmf/nmf.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(NMFTest);

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::nmf;

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix.
 */
BOOST_AUTO_TEST_CASE(NMFTest)
{
  mat v = randu<mat>(5, 5);
  size_t r = 4;
  mat w, h;

  NMF<> nmf;
  nmf.Apply(V, W, H, r);

  mat wh = w * h;

  for (size_t row = 0; row < 5; row++)
    for (size_t col = 0; col < 5; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 5);
}


BOOST_AUTO_TEST_SUITE_END();
