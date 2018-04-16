/**
 * @file spsa_test.cpp
 * @author N Rajiv Vaidyanathan
 *
 * Test file for SPSA (simultaneous pertubation stochastic approximation).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/spsa/spsa.hpp>
#include <mlpack/core/optimizers/problems/sphere_function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(SPSATest);

/**
 * Tests the SPSA optimizer using the Sphere Function.
 */
BOOST_AUTO_TEST_CASE(SimpleSPSATestFunction)
{
  for (size_t i = 10; i <= 50; i++)
  {
    SphereFunction h(i);

    SPSA optimiser(0.1, 0.102, 0.16,
                   0.3, 100000);

    arma::mat coordinates = h.GetInitialPoint();
    double result = optimiser.Optimize(h, coordinates);

    if (i <= 30)
      BOOST_REQUIRE_CLOSE(result, 0.0, 10000);
    else if (i <= 34 || i == 36 || i == 39)
      BOOST_REQUIRE_CLOSE(result, 8e-33, 10000);
    else
      BOOST_REQUIRE_CLOSE(result, 3e-32, 100000);

    BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
    BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
    BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
  }
}

BOOST_AUTO_TEST_SUITE_END();
