/**
 * @file aug_lagrangian_test.cpp
 * @author Ryan Curtin
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.hpp.
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian_test_functions.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

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
