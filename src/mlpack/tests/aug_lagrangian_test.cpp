/***
 * @file aug_lagrangian_test.cc
 * @author Ryan Curtin
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.h.
 */

#include <mlpack/core.h>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian_test_functions.hpp>
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(AugLagrangianTest);

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
   */
  BOOST_AUTO_TEST_CASE(extremely_simple_lovasz_theta_sdp) {
    // Manually input the single edge.
    arma::mat edges = "0; 1";

    LovaszThetaSDP ltsdp(edges);
    AugLagrangian<LovaszThetaSDP> aug(ltsdp, 10);

    arma::mat coords = ltsdp.GetInitialPoint();

    if (!aug.Optimize(0, coords))
      BOOST_FAIL("Optimization reported failure.");

    double final_value = ltsdp.Evaluate(coords);

    arma::mat X = trans(coords) * coords;

    BOOST_CHECK_CLOSE(final_value, -1.0, 1e-5);

    BOOST_CHECK_CLOSE(X(0, 0) + X(1, 1), 1.0, 1e-5);
    BOOST_CHECK_SMALL(X(0, 1), 1e-8);
    BOOST_CHECK_SMALL(X(1, 0), 1e-8);
  }

  /***
   * Tests the Augmented Lagrangian optimizer on the Lovasz theta SDP, using the
   * hamming_10_2 dataset, just like in the paper by Monteiro and Burer.
   *
  BOOST_AUTO_TEST_CASE(lovasz_theta_johnson8_4_4) {
    arma::mat edges;
    // Hardcoded filename: bad!
    data::Load("MANN-a27.csv", edges);

    LovaszThetaSDP ltsdp(edges);
    AugLagrangian<LovaszThetaSDP> aug(ltsdp, 10);

    arma::mat coords = ltsdp.GetInitialPoint();

    if(!aug.Optimize(0, coords))
      BOOST_FAIL("Optimization reported failure.");

    double final_value = ltsdp.Evaluate(coords);

    BOOST_REQUIRE_CLOSE(final_value, -14.0, 1e-5);
  }
  */
BOOST_AUTO_TEST_SUITE_END();
