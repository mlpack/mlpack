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

/***
 * Extremely simple test case for the Lovasz theta SDP.
 *
BOOST_AUTO_TEST_CASE(extremely_simple_lovasz_theta_sdp) {
  arma::mat edges = "0; 1";

  LovaszThetaSDP ltsdp(edges);
  AugLagrangian<LovaszThetaSDP> aug(ltsdp, 10);

  arma::mat coords = ltsdp.GetInitialPoint();

  if (!aug.Optimize(0, coords))
    BOOST_FAIL("Optimization reported failure.");

  double final_value = ltsdp.Evaluate(coords);

//  BOOST_REQUIRE_CLOSE(final_value, 1.0, 1e-5);

  arma::mat X = trans(coords) * coords;

  std::cout << X;

  BOOST_REQUIRE_CLOSE(X(0, 0), 0.5, 1e-5);
  BOOST_REQUIRE_CLOSE(X(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(X(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(X(1, 1), 0.5, 1e-5);
}

/***
 * Tests the Augmented Lagrangian optimizer on the Lovasz theta SDP, using the
 * hamming_10_2 dataset, just like in the paper by Monteiro and Burer.
 *
BOOST_AUTO_TEST_CASE(lovasz_theta_hamming_10_2) {
  arma::mat edges;
  // Hardcoded filename: bad!
  data::Load("johnson8-4-4.csv", edges);

  LovaszThetaSDP ltsdp(edges);
  AugLagrangian<LovaszThetaSDP> aug(ltsdp, 10);

  arma::mat coords = ltsdp.GetInitialPoint();

  IO::Debug << "Optimizing..." << std::endl;

  if(!aug.Optimize(0, coords))
    BOOST_FAIL("Optimization reported failure.");

  double final_value = ltsdp.Evaluate(coords);

  BOOST_REQUIRE_CLOSE(final_value, -14.0, 1e-5);
}
*/
