/**
 * @file tests/xgboost_test.cpp
 * @author Rishabh Garg
 *
 * Tests for the XGBoost class and related classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/xgboost/loss_functions/sse_loss.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;

/**
 * Test that the initial prediction is calculated correctly for SSE loss.
 */
TEST_CASE("SSEInitialPredictionTest", "[XGBTest]")
{
  arma::vec values = {1, 3, 2, 2, 5, 6, 9, 11, 8, 8};

  double initPred = 5.5;

  SSELoss Loss;
  REQUIRE(Loss.InitialPrediction(values) == initPred);
}

/**
 * Test that output leaf value is calculated correctly for SSE Loss.
 */
TEST_CASE("SSELeafValueTest", "[XGBTest]")
{
  arma::mat input = { { 1,   3,   2,   2, 5, 6, 9,    11, 8,   8 },
                      { 0.5, 1, 2.5, 1.5, 5, 8, 8, 10.75, 9, 9.5 } };
  arma::vec weights; // dummy weights not used.

  // Actual output leaf value.
  double leafValue = -0.075;

  SSELoss Loss;
  (void) Loss.Evaluate<false>(input, weights);

  REQUIRE(Loss.OutputLeafValue(input, weights) == leafValue);
}

/**
 * Test that the gain is computed correctly for SSE Loss.
 */
TEST_CASE("SSEGainTest", "[XGBTest]")
{
  arma::mat input = { { 1,   3,   2,   2, 5, 6, 9,    11, 8,   8 },
                      { 0.5, 1, 2.5, 1.5, 5, 8, 8, 10.75, 9, 9.5 } };
  arma::vec weights; // dummy weights not used.

  // Actual gain value.
  double gain = 0.05625;

  SSELoss Loss;
  REQUIRE(Loss.Evaluate<false>(input, weights) == gain);
}
