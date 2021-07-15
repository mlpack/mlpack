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
using namespace mlpack::ensemble;

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
 * Test that gradients are calculated correctly for SSE Loss.
 */
TEST_CASE("SSEGradientsTest", "[XGBTest]")
{
  arma::vec observed = {1, 3, 2, 2, 5, 6, 9, 11, 8, 8};
  arma::vec predicted = {0.5, 1, 2.5, 1.5, 5, 8, 8, 10.75, 9, 9.5};

  // Actual gradients.
  arma::vec gradients = {-0.5, -2, 0.5, -0.5, 0, 2, -1, -0.25, 1, 1.5};

  SSELoss Loss;
  // Calculated gradients.
  arma::vec calculatedGradients = Loss.Gradients(observed, predicted);

  for (int i = 0; i < 10; i++)
    REQUIRE(calculatedGradients[i] == gradients[i]);
}

/**
 * Test that hessians are calculated correctly for SSE Loss.
 */
TEST_CASE("SSEHessiansTest", "[XGBTest]")
{
  arma::vec observed = {1, 3, 2, 2, 5, 6, 9, 11, 8, 8};
  arma::vec predicted = {0.5, 1, 2.5, 1.5, 5, 8, 8, 10.75, 9, 9.5};

  // Actual hessians.
  arma::vec hessians = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  SSELoss Loss;
  // Calculated hessians.
  arma::vec calculatedHessians = Loss.Hessians(observed, predicted);

  for (int i = 0; i < 10; i++)
    REQUIRE(calculatedHessians[i] == hessians[i]);
}

/**
 * Test that residuals are calculated correctly for SSE Loss.
 */
TEST_CASE("SSEResidualsTest", "[XGBTest]")
{
  arma::vec observed = {1, 3, 2, 2, 5, 6, 9, 11, 8, 8};
  arma::vec predicted = {0.5, 1, 2.5, 1.5, 5, 8, 8, 10.75, 9, 9.5};

  // Actual residuals.
  arma::vec residuals = {0.5, 2, -0.5, 0.5, 0, -2, 1, 0.25, -1, -1.5};

  SSELoss Loss;
  // Calculated residuals.
  arma::vec calculatedResiduals = Loss.Residuals(observed, predicted);

  for (int i = 0; i < 10; i++)
    REQUIRE(calculatedResiduals[i] == residuals[i]);
}

/**
 * Test that output leaf value is calculated correctly for SSE Loss.
 */
TEST_CASE("SSELeafValueTest", "[XGBTest]")
{
  arma::vec observed = {1, 3, 2, 2, 5, 6, 9, 11, 8, 8};
  arma::vec predicted = {0.5, 1, 2.5, 1.5, 5, 8, 8, 10.75, 9, 9.5};

  // Actual output leaf value.
  double leafValue = -0.075;

  SSELoss Loss;
  // Calculating gradients and hessians for input to OutputLeafValue().
  arma::vec gradients = Loss.Gradients(observed, predicted);
  arma::vec hessians = Loss.Hessians(observed, predicted);

  REQUIRE(Loss.OutputLeafValue(gradients, hessians) == leafValue);
}
