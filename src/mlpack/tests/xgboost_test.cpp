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
#include <mlpack/methods/xgboost/xgb_exact_numeric_split.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::ensemble;

/**
 * Test that the initial prediction is calculated correctly for SSE loss.
 */
TEST_CASE("SSEInitialPredictionTest", "[XGBTest]")
{
  arma::vec values = {1, 3, 2, 2, 5, 6, 9, 11, 8, 8};

  double initPred = 5.5;

  REQUIRE(SSELoss::InitialPrediction(values) == initPred);
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

/**
 * Make sure the SSE loss is zero when the responses are perfect.
 */
TEST_CASE("SSELossPerfectTest", "[XGBTest]")
{
  arma::mat input = { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } };
  arma::vec weights; // dummy weights not used.

  SSELoss Loss;
  REQUIRE(Loss.Evaluate<false>(input, weights) == Approx(0.0).margin(1e-5));
}

/**
 * The SSe loss of an empty input is 0.
 */
TEST_CASE("SSELossEmptyTest", "[XGBTest]")
{
  arma::mat input;
  arma::vec weights; // dummy weights not used.

  SSELoss Loss;
  REQUIRE(Loss.Evaluate<false>(input, weights) == Approx(0.0).margin(1e-5));
}

/**
 * Check that the XGBExactNumericSplit will split on an obviously splittable
 * dimension.
 */
TEST_CASE("XGBExactNumericSplitSimpleSplitTest", "[XGBTest]")
{
  arma::rowvec predictors = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  arma::mat input = { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                      { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 } };
  arma::vec weights; // dummy weights not used.

  SSELoss Loss;
  double splitInfo;
  XGBExactNumericSplit<SSELoss>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = Loss.Evaluate<false>(input, weights);
  const double gain = XGBExactNumericSplit<SSELoss>::SplitIfBetter<false>(
      bestGain, predictors, input, weights, 0, 0, splitInfo, aux, Loss);

  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  // Make sure that the split was between 5 and 6.
  REQUIRE(splitInfo > 5);
  REQUIRE(splitInfo < 6);
}

/**
 * Check that the XGBExactNumericSplit won't split if both left and right
 * possible childs violate the minChildWeight condition.
 */
TEST_CASE("XGBExactNumericSplitMinChildWeightTest", "[XGBTest]")
{
  arma::rowvec predictors = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  arma::mat input = { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                      { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 } };
  arma::vec weights; // dummy weights not used.

  SSELoss Loss(0, 0, 6); // minChildWeight = 6
  double splitInfo;
  XGBExactNumericSplit<SSELoss>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = Loss.Evaluate<false>(input, weights);
  const double gain = XGBExactNumericSplit<SSELoss>::SplitIfBetter<false>(
      bestGain, predictors, input, weights, 0, 0, splitInfo, aux, Loss);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
}
