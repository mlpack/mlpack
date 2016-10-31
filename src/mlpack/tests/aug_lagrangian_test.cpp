/**
 * @file aug_lagrangian_test.cpp
 * @author Ryan Curtin
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.hpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian_test_functions.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(AugLagrangianTest);

/**
 * Tests the Augmented Lagrangian optimizer using the
 * AugmentedLagrangianTestFunction class.
 */
BOOST_AUTO_TEST_CASE(AugLagrangianTestFunctionTest)
{
  // The choice of 10 memory slots is arbitrary.
  AugLagrangianTestFunction f;
  AugLagrangian<AugLagrangianTestFunction> aug(f);

  arma::vec coords = f.GetInitialPoint();

  if (!aug.Optimize(coords, 0))
    BOOST_FAIL("Optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  BOOST_REQUIRE_CLOSE(finalValue, 70.0, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[0], 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[1], 4.0, 1e-5);
}

/**
 * Tests the Augmented Lagrangian optimizer using the Gockenbach function.
 */
BOOST_AUTO_TEST_CASE(GockenbachFunctionTest)
{
  GockenbachFunction f;
  AugLagrangian<GockenbachFunction> aug(f);

  arma::vec coords = f.GetInitialPoint();

  if (!aug.Optimize(coords, 0))
    BOOST_FAIL("Optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  // Higher tolerance for smaller values.
  BOOST_REQUIRE_CLOSE(finalValue, 29.633926, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[0], 0.12288178, 1e-3);
  BOOST_REQUIRE_CLOSE(coords[1], -1.10778185, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[2], 0.015099932, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();

