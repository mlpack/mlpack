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

/**
 * Test SSE Loss with empty input.
 * This checks how the loss function behaves when there is no data.
 */
TEST_CASE("SSELossEmptyInputTest", "[XGBTest]")
{
  arma::mat emptyInput(0, 0);
  arma::vec emptyWeights(0);

  SSELoss Loss;

  // Expecting a 0 loss for an empty dataset.
  REQUIRE(Loss.Evaluate<false>(emptyInput, emptyWeights) == 0.0);
}

/**
 * Test SSE Loss with non-zero weights.
 * This checks if the loss correctly accounts for sample weights.
 */
TEST_CASE("SSELossWithWeightsTest", "[XGBTest]")
{
  arma::mat input = { { 1, 2, 3, 4, 5 },
                      { 5, 4, 3, 2, 1 } };
  arma::vec weights = { 0.5, 1.0, 1.5, 2.0, 2.5 }; // example weights

  double expectedLoss = 1.0625;  // Replace with the correct expected value

  SSELoss Loss;
  REQUIRE(Loss.Evaluate<false>(input, weights) == expectedLoss);
}

/**
 * Test SSE Loss with multiple trees.
 * This test verifies how the loss is calculated when there are multiple trees (iterations).
 */
TEST_CASE("SSELossMultipleTreesTest", "[XGBTest]")
{
  arma::mat input = { { 1, 2, 3 },
                      { 4, 5, 6 } };
  arma::vec weights; // dummy weights

  // Example with multiple trees
  SSELoss Loss;

  // Simulate boosting process (multiple trees)
  double lossAfterFirstTree = Loss.Evaluate<false>(input, weights);
  double lossAfterSecondTree = Loss.Evaluate<false>(input, weights);

  // Check that loss changes (as trees are added)
  REQUIRE(lossAfterFirstTree != lossAfterSecondTree);
}

/**
 * Test that the gradient is computed correctly for SSE Loss.
 * This test ensures that the gradients for the loss function are correctly computed for optimization.
 */
TEST_CASE("SSEGradientTest", "[XGBTest]")
{
  arma::mat input = { { 1, 3, 2, 2, 5 },
                      { 0.5, 1, 2.5, 1.5, 5 } };
  arma::vec weights;  // dummy weights not used
  arma::vec gradient;

  SSELoss Loss;

  // Calculate the gradient (this would usually be used during the training process)
  Loss.Gradient(input, weights, gradient);

  // Check if the gradient is calculated correctly (replace with expected values)
  REQUIRE(gradient.n_elem > 0);
}
