/**
 * @file loss_functions_test.cpp
 * @author Dakshit Agrawal
 *
 * Tests for loss functions in mlpack::methods::ann:loss_functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/methods/ann/loss_functions/KLDivergence.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(LossFunctionsTest);

/**
 * Simple KL Divergence test.  The loss should be zero if input = target.
 */
BOOST_AUTO_TEST_CASE(SimpleKLDivergenceTest)
{
  arma::mat input, target, output;
  double loss;
  KLDivergence<> module(true);

  // Test the Forward function.  Loss should be 0 if input = target.
  input = arma::ones(10, 1);
  target = arma::ones(10, 1);
  loss = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_SMALL(loss, 0.00001);
}

/**
 * Test to check KL Divergence loss function when we take mean.
 */

BOOST_AUTO_TEST_CASE(KLDivergenceMeanTest)
{
  arma::mat input, target, output;
  double loss;
  KLDivergence<> module(true);

  // Test the Forward function.
  input = arma::mat("1 1 1 1 1 1 1 1 1 1");
  target = arma::exp(arma::mat("2 1 1 1 1 1 1 1 1 1"));

  loss = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_CLOSE_FRACTION(loss, -1.1 , 0.00001);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(target), std::move(output));
  BOOST_REQUIRE_CLOSE_FRACTION(arma::as_scalar(output), -0.1, 0.00001);
}

/**
 * Test to check KL Divergence loss function when we do not take mean.
 */

BOOST_AUTO_TEST_CASE(KLDivergenceNoMeanTest)
{
  arma::mat input, target, output;
  double loss;
  KLDivergence<> module(false);

  // Test the Forward function.
  input = arma::mat("1 1 1 1 1 1 1 1 1 1");
  target = arma::exp(arma::mat("2 1 1 1 1 1 1 1 1 1"));

  loss = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_CLOSE_FRACTION(loss, -11, 0.00001);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(target), std::move(output));
  BOOST_REQUIRE_CLOSE_FRACTION(arma::as_scalar(output), -1, 0.00001);
}


BOOST_AUTO_TEST_SUITE_END();
