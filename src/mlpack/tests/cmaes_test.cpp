/**
 * @file sgd_test.cpp
 * @author Ryan Curtin
 *
 * Test file for SGD (stochastic gradient descent).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/cmaes/cmaes.hpp>
#include <mlpack/core/optimizers/cmaes/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(CMAESTest);

BOOST_AUTO_TEST_CASE(SimpleCMAESTestFunction)
{
  cmaesTestFunction test;

  ssize_t N = test.NumFunctions();

  arma::mat start(N, 1); start.fill(0.5);
  arma::mat initialStdDeviations(N, 1); initialStdDeviations.fill(1.5);

  CMAES<cmaesTestFunction> s(test, start, initialStdDeviations, 10000, 1e-18);

  arma::mat coordinates(N, 1);
  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}

BOOST_AUTO_TEST_SUITE_END();


