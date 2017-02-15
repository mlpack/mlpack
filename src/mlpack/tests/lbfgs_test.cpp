/**
 * @file lbfgs_test.cpp
 *
 * Tests the L-BFGS optimizer on a couple test functions.
 *
 * @author Ryan Curtin (gth671b@mail.gatech.edu)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(LBFGSTest);

/**
 * Tests the L-BFGS optimizer using the Rosenbrock Function.
 */
BOOST_AUTO_TEST_CASE(RosenbrockFunctionTest)
{
  RosenbrockFunction f;
  L_BFGS<RosenbrockFunction> lbfgs(f);
  lbfgs.MaxIterations() = 10000;

  arma::vec coords = f.GetInitialPoint();
  if (!lbfgs.Optimize(coords))
    BOOST_FAIL("L-BFGS optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  BOOST_REQUIRE_SMALL(finalValue, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[0], 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[1], 1.0, 1e-5);
}

/**
 * Tests the L-BFGS optimizer using the Wood Function.
 */
BOOST_AUTO_TEST_CASE(WoodFunctionTest)
{
  WoodFunction f;
  L_BFGS<WoodFunction> lbfgs(f);
  lbfgs.MaxIterations() = 10000;

  arma::vec coords = f.GetInitialPoint();
  if (!lbfgs.Optimize(coords))
    BOOST_FAIL("L-BFGS optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  BOOST_REQUIRE_SMALL(finalValue, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[0], 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[1], 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[2], 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[3], 1.0, 1e-5);
}

/**
 * Tests the L-BFGS optimizer using the generalized Rosenbrock function.  This
 * is actually multiple tests, increasing the dimension by powers of 2, from 4
 * dimensions to 1024 dimensions.
 */
BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockFunctionTest)
{
  for (int i = 2; i < 10; i++)
  {
    // Dimension: powers of 2
    int dim = std::pow(2.0, i);

    GeneralizedRosenbrockFunction f(dim);
    L_BFGS<GeneralizedRosenbrockFunction> lbfgs(f, 20);
    lbfgs.MaxIterations() = 10000;

    arma::vec coords = f.GetInitialPoint();
    if (!lbfgs.Optimize(coords))
      BOOST_FAIL("L-BFGS optimization reported failure.");

    double finalValue = f.Evaluate(coords);

    // Test the output to make sure it is correct.
    BOOST_REQUIRE_SMALL(finalValue, 1e-5);
    for (int j = 0; j < dim; j++)
      BOOST_REQUIRE_CLOSE(coords[j], 1.0, 1e-5);
  }
}

/**
 * Tests the L-BFGS optimizer using the Rosenbrock-Wood combined function.  This
 * is a test on optimizing a matrix of coordinates.
 */
BOOST_AUTO_TEST_CASE(RosenbrockWoodFunctionTest)
{
  RosenbrockWoodFunction f;
  L_BFGS<RosenbrockWoodFunction> lbfgs(f);
  lbfgs.MaxIterations() = 10000;

  arma::mat coords = f.GetInitialPoint();
  if (!lbfgs.Optimize(coords))
    BOOST_FAIL("L-BFGS optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  BOOST_REQUIRE_SMALL(finalValue, 1e-5);
  for (int row = 0; row < 4; row++)
  {
    BOOST_REQUIRE_CLOSE((coords(row, 0)), 1.0, 1e-5);
    BOOST_REQUIRE_CLOSE((coords(row, 1)), 1.0, 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
