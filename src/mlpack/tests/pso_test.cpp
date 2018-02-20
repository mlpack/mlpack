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
#include <mlpack/core/optimizers/problems/matyas_function.cpp>
#include <mlpack/core/optimizers/problems/booth_function.cpp>
#include <mlpack/core/optimizers/problems/wood_function.cpp>
#include <mlpack/core/optimizers/problems/mc_cormick_function.cpp>
#include <mlpack/core/optimizers/problems/eggholder_function.cpp>

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
BOOST_AUTO_TEST_CASE(RosenbrockFunctionTest)
{
  PSO optimizer;
  RosenbrockFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-5);
}

/**
 * Test for minimizing Matyas function.
 */
BOOST_AUTO_TEST_CASE(MatyasFunctionTest)
{
  PSO optimizer;
  MatyasFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-5);
}

/**
 * Test for minimizing the Booth function.
 */
BOOST_AUTO_TEST_CASE(BoothFunctionTest)
{
  PSO optimizer(60, 0.9, 0.5, 0.3, 200, 1e-3);
  BoothFunction f;

  arma::mat iterate;

  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-3);
}

/**
 * Test for the McCormick function.
 */
BOOST_AUTO_TEST_CASE(McCormickTest)
{
  PSO optimizer;
  McCormickFunction f;

  arma::mat iterate;
  iterate << -0.54719 << -1.54719;

  double result = optimizer.Optimize(f, iterate);

  BOOST_TEST(result == -1.9133, boost::test_tools::tolerance(1e-3));
}

/**
 * Test for the Eggholder function.
 */
BOOST_AUTO_TEST_CASE(EggholderFunctionTest)
{
  PSO optimizer;
  EggholderFunction f;

  arma::mat iterate;
  iterate << 512 << 404.2319;

  double result = optimizer.Optimize(f, iterate);

  BOOST_TEST(result == -959.6407, boost::test_tools::tolerance(1e-3));
}

BOOST_AUTO_TEST_SUITE_END();
