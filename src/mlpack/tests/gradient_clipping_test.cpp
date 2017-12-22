/**
 * @file gradient_clipping_test.cpp
 * @author Konstantin Sidorov
 *
 * Test file for gradient clipping.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/gradient_clipping.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/momentum_update.hpp>
#include <mlpack/core/optimizers/problems/sgd_test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(GradientClippingTest);

// Test checking that gradient clipping works with vanilla update.
BOOST_AUTO_TEST_CASE(ClippedVanillaUpdateTest)
{
  VanillaUpdate vanillaUpdate;
  GradientClipping<VanillaUpdate> update(-3.0, +3.0, vanillaUpdate);
  update.Initialize(3, 3);

  arma::mat coordinates = arma::zeros(3, 3);
  // Setting step = 1 to make math easy.
  double stepSize = 1.0;
  arma::mat dummyGradient("-6 6 0; 1 2 3; -3 0 4;");
  update.Update(coordinates, stepSize, dummyGradient);
  // After clipping, we should get the following coordinates:
  arma::mat targetCoordinates("3 -3 0; -1 -2 -3; 3 0 -3;");
  BOOST_REQUIRE_SMALL(arma::abs(coordinates - targetCoordinates).max(), 1e-7);
}

// Test checking that gradient clipping works with momentum update.
BOOST_AUTO_TEST_CASE(ClippedMomentumUpdateTest)
{
  // Once again, setting momentum = 1 for easy math
  // (now momentum = -stepSize * [sum of gradients])
  MomentumUpdate momentumUpdate(1);
  GradientClipping<MomentumUpdate> update(-3.0, +3.0, momentumUpdate);
  update.Initialize(3, 3);

  arma::mat coordinates = arma::zeros(3, 3);
  double stepSize = 1.0;
  arma::mat dummyGradient("-6 6 0; 1 2 3; -3 0 4;");
  update.Update(coordinates, stepSize, dummyGradient);
  arma::mat targetCoordinates("3 -3 0; -1 -2 -3; 3 0 -3;");
  // On the first Update() call the parameters
  // should just be equal to (-gradient).
  BOOST_REQUIRE_SMALL(arma::abs(coordinates - targetCoordinates).max(), 1e-7);
  update.Update(coordinates, stepSize, dummyGradient);
  // On the second Update() call the Momentum update will subtract
  // the gradient from the momentum, which gives 2 * gradient value
  // for the momentum on that step. Adding that to the gradient which
  // was subtracted earlier yiels the 3 * gradient in the following check.
  BOOST_REQUIRE_SMALL(
    arma::abs(coordinates - 3 * targetCoordinates).max(), 1e-7);
}

BOOST_AUTO_TEST_SUITE_END();
