/***
 * @file aug_lagrangian_test.cc
 * @author Ryan Curtin
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.h.
 */

#include <fastlib/fastlib.h>
#include "aug_lagrangian.h"
#include "aug_lagrangian_test_functions.h"

#define BOOST_TEST_MODULE Augmented Lagrangian Test
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::optimization;

/***
 * Tests the Augmented Lagrangian optimizer using the
 * AugmentedLagrangianTestFunction class.
 */
BOOST_AUTO_TEST_CASE(aug_lagrangian_test_function) {
  // The choice of 10 memory slots is arbitrary.
  AugLagrangianTestFunction f;
  AugLagrangian<AugLagrangianTestFunction> aug(f, 10);

  arma::vec coords = f.GetInitialPoint();

  if(!aug.Optimize(0, coords))
    BOOST_FAIL("Optimization reported failure.");
  
  double final_value = f.Evaluate(coords);

  BOOST_REQUIRE_CLOSE(final_value, 70, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[0], 1, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[1], 4, 1e-5);
}

/***
 * Tests the Augmented Lagrangian optimizer using the Gockenbach function.
 */
BOOST_AUTO_TEST_CASE(gockenbach_function) {
  GockenbachFunction f;
  AugLagrangian<GockenbachFunction> aug(f, 10);

  arma::vec coords = f.GetInitialPoint();

  if(!aug.Optimize(0, coords))
    BOOST_FAIL("Optimization reported failure.");

  double final_value = f.Evaluate(coords);

  BOOST_REQUIRE_CLOSE(final_value, 29.633926, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[0], 0.12288178, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[1], -1.10778185, 1e-5);
  BOOST_REQUIRE_CLOSE(coords[2], 0.015099932, 1e-5);
}
