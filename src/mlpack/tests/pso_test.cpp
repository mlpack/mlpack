/**
 * @file pso_test.cpp
 * @author Adeel Ahmad
 *
 * Test file for PSO.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/pso/pso.hpp>
#include <mlpack/core/optimizers/pso/test_function.hpp>
#include <mlpack/core/optimizers/problems/rosenbrock_function.cpp>

#include <boost/test/unit_test.hpp>

using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(PSOTest);

/**
 * Simple test case for PSO.
 */
BOOST_AUTO_TEST_CASE(SimpleTest)
{
  PSO optimizer;
  PSOTestFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-5);
}

/**
 * Test for minimizing the Rosenbrock function.
 */
BOOST_AUTO_TEST_CASE(RosenbrockTest)
{
  PSO optimizer;
  RosenbrockFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
