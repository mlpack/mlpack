/**
 * @file tests/decision_tree_regressor_test.cpp
 * @author Rishabh Garg
 *
 * Tests for the DecisionTreeRegressor class and related classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/mad_gain.hpp>
#include <mlpack/methods/decision_tree/mse_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::distribution;

/**
 * Make sure the MSE gain is zero when the labels are perfect.
 */
TEST_CASE("MSEGainPerfectTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec labels;
  labels.ones(10);

  REQUIRE(MSEGain::Evaluate<false>(labels, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Make sure that the MSE gain is equal to negative of variance.
 */
TEST_CASE("MSEGainVarianceTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights(100, arma::fill::ones);
  arma::rowvec labels(100, arma::fill::randn);

  // Theoretical gain.
  double theoreticalGain = - arma::var(labels) * 99.0 / 100.0;

  // Calculated gain.
  const double calculatedGain = MSEGain::Evaluate<false>(labels, weights);

  REQUIRE(calculatedGain == Approx(theoreticalGain).margin(1e-9));
}

/**
 * The MSE gain of an empty vector is 0.
 */
TEST_CASE("MSEGainEmptyTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec labels;
  REQUIRE(MSEGain::Evaluate<false>(labels, weights) ==
          Approx(0.0).margin(1e-5));

  REQUIRE(MSEGain::Evaluate<true>(labels, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Make sure the MAD gain is zero when the labels are perfect.
 */
TEST_CASE("MADGainPerfectTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec labels;
  labels.ones(10);

  REQUIRE(MADGain::Evaluate<false>(labels, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Make sure that when mean of labels is zero, MAD_gain = mean of
 * absolute values of the distribution.
 */
TEST_CASE("MADGainNormalTest", "[DecisionTreeRegressionTest")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec labels = { 1, 2, 3, 4, 5, -1, -2, -3, -4, -5 }; // Mean = 0.

  // Theoretical gain.
  double theoreticalGain = 0.0;
  for (size_t i = 0; i < labels.n_elem; ++i)
    theoreticalGain -= std::abs(labels[i]);
  theoreticalGain /= (double) labels.n_elem;

  // Calculated gain.
  const double calculatedGain = MADGain::Evaluate<false>(labels, weights);

  REQUIRE(calculatedGain == Approx(theoreticalGain).margin(1e-5));
}

/**
 * The MAD gain of an empty vector is 0.
 */
TEST_CASE("MADGainEmptyTest", "[DecisionTreeRegressionTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec labels;
  REQUIRE(MADGain::Evaluate<false>(labels, weights) ==
          Approx(0.0).margin(1e-5));

  REQUIRE(MADGain::Evaluate<true>(labels, weights) ==
          Approx(0.0).margin(1e-5));
}
