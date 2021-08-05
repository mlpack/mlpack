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
#include <mlpack/methods/xgboost/xgboost_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "test_function_tools.hpp"

using namespace mlpack;
using namespace mlpack::ensemble;
using namespace mlpack::tree;

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

  REQUIRE(Loss.OutputLeafValue<false>(input, weights) == leafValue);
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

/**
 * Make sure an empty forest cannot predict.
 */
TEST_CASE("EmptyPredictTest", "[XGBTest]")
{
  XGBoostTreeRegressor<> xgb; // No training.

  arma::mat points(10, 100, arma::fill::randu);
  arma::rowvec predictions;

  REQUIRE_THROWS_AS(xgb.Predict(points.col(0)), std::invalid_argument);
  REQUIRE_THROWS_AS(xgb.Predict(points, predictions), std::invalid_argument);
}

/**
 * A basic construction of the xgboost---ensure that a forest of trees is
 * generated.
 */
TEST_CASE("BasicConstructionTestXGB", "[XGBTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::rowvec responses(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 1.0;
  }

  // Use default parameters.
  XGBoostTreeRegressor<> xgb(dataset, responses);

  REQUIRE(xgb.NumTrees() == 100); // 100 is the default value.
}

/**
 * Test unweighted numeric learning, making sure that we get better
 * performance than a single decision tree.
 */
TEST_CASE("NumericalLearningTest", "[XGBTest]")
{
  arma::mat X;
  arma::rowvec Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  arma::mat XTrain, XTest;
  arma::rowvec YTrain, YTest;
  data::Split(X, Y, XTrain, XTest, YTrain, YTest, 0.3);

  XGBoostTreeRegressor<> xgb(XTrain, YTrain);
  DecisionTreeRegressor<> dt(XTrain, YTrain, 5);

  // Making predicstions using xgboost.
  arma::rowvec xgbPredictions, dtPredictions;
  xgb.Predict(XTest, xgbPredictions);
  const double xgbRMSE = RMSE(xgbPredictions, YTest);

  // Making predictions using decision tree.
  dt.Predict(XTest, dtPredictions);
  const double treeRMSE = RMSE(dtPredictions, YTest);

  REQUIRE(xgbRMSE < treeRMSE);
}

/**
 * Test that different trees get generated.
 */
TEST_CASE("DifferentTreesTestXGB", "[XGBTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::rowvec responses(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 1.0;
  }

  bool success = false;
  size_t trial = 0;

  // It's possible we might get the same random dimensions selected, so let's do
  // multiple trials.
  while (!success && trial < 5)
  {
    XGBoostTreeRegressor<SSELoss, RandomDimensionSelect> xgb;
    xgb.Train(dataset, responses, 2);

    success = (xgb.Tree(0).SplitDimension() != xgb.Tree(1).SplitDimension());

    ++trial;
  }

  REQUIRE(success == true);
}

/**
 * Test that XGBoostTreeRegressor::Train() when passed warmStart = True
 * trains on top of exixting forest and adds the newly trained trees to
 * the previously exixting forest.
 */
TEST_CASE("WarmStartTreesTestXGB", "[XGBTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::rowvec responses(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 1.0;
  }

  // Train a xgboost regressor.
  XGBoostTreeRegressor<> xgb(dataset, responses, 25 /* 25 trees */);

  REQUIRE(xgb.NumTrees() == 25);

  xgb.Train(dataset, responses, 20 /* 20 trees */, 6, 1, 0, 1, 0,
      true /* warmStart */);

  REQUIRE(xgb.NumTrees() == 25 + 20);
}

/**
 * Test that XGBoostTreeRegressor::Train() when passed warmStart = True
 * does not drop prediction quality on train data. Note that prediction
 * quality on test data may drop due to overfitting in some cases.
 */
TEST_CASE("WarmStartPredictionsQualityTestXGB", "[XGBTest]")
{
  arma::mat X;
  arma::rowvec Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  XGBoostTreeRegressor<> xgb(X, Y);

  // Get performance statistics on train data.
  arma::rowvec oldPredictions;
  xgb.Predict(X, oldPredictions);
  const double oldRMSE = RMSE(oldPredictions, Y);

  // Fitting more trees on top of existing ones.
  xgb.Train(X, Y, 25, 6, 1, 0, 1, 0, true /* warmStart */);

  // Get performance statistics on train data.
  arma::rowvec newPredictions;
  xgb.Predict(X, newPredictions);
  const double newRMSE = RMSE(newPredictions, Y);

  REQUIRE(xgb.NumTrees() == 100 + 25);
  REQUIRE(newRMSE <= oldRMSE);
}
