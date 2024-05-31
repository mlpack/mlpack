/**
 * @file tests/adaboost_regressor_test.cpp
 * @author Dinesh Kumar
 *
 * Tests for the AdaBoostRegressor<> class and LossFunctions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/adaboost.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"
#include "test_function_tools.hpp"

using namespace mlpack;

/**
 * Creates dataset with 5 groups with all the points in same group have exactly
 * same responses.
 */
void CreateMultipleSplitData(arma::mat& d, arma::rowvec& r, const size_t count,
    arma::rowvec& values)
{
  d = arma::mat(10, count, arma::fill::randu);
  r = arma::rowvec(count);

  // Group 1.
  for (size_t i = 0; i < count / 5; i++)
  {
    d(3, i) = i;
    r(i) = values[0];
  }
  // Group 2.
  for (size_t i = count / 5; i < (count / 5) * 2; i++)
  {
    d(3, i) = i;
    r(i) = values[1];
  }
  // Group 3.
  for (size_t i = (count / 5) * 2; i < (count / 5) * 3; i++)
  {
    d(3, i) = i;
    r(i) = values[2];
  }
  // Group 4.
  for (size_t i = (count / 5) * 3; i < (count / 5) * 4; i++)
  {
    d(3, i) = i;
    r(i) = values[3];
  }
  // Group 5.
  for (size_t i = (count / 5) * 4; i < count; i++)
  {
    d(3, i) = i;
    r(i) = values[4];
  }
}

void CreateMultipleSplitFloatData(arma::fmat& d, arma::frowvec& r, const size_t count,
    arma::frowvec& values)
{
  d = arma::fmat(10, count, arma::fill::randu);
  r = arma::frowvec(count);

  // Group 1.
  for (size_t i = 0; i < count / 5; i++)
  {
    d(3, i) = i;
    r(i) = values[0];
  }
  // Group 2.
  for (size_t i = count / 5; i < (count / 5) * 2; i++)
  {
    d(3, i) = i;
    r(i) = values[1];
  }
  // Group 3.
  for (size_t i = (count / 5) * 2; i < (count / 5) * 3; i++)
  {
    d(3, i) = i;
    r(i) = values[2];
  }
  // Group 4.
  for (size_t i = (count / 5) * 3; i < (count / 5) * 4; i++)
  {
    d(3, i) = i;
    r(i) = values[3];
  }
  // Group 5.
  for (size_t i = (count / 5) * 4; i < count; i++)
  {
    d(3, i) = i;
    r(i) = values[4];
  }
}

/**
 * Make sure the LinearLoss is of all elements is 1, when the elements are 
 * all equal.
 */
TEST_CASE("LinearLossEqualTest", "[AdaBoostRegressorTest]")
{
  arma::rowvec values(10);
  values.fill(5);
  arma::rowvec loss = LinearLoss::Calculate(values);
  for(size_t i = 0; i < 10; i++)
    REQUIRE(loss[i] == Approx(1.0).margin(1e-5));
}

/**
 * Make sure the SquareLoss is of all elements is 1, when the elements are 
 * all equal.
 */
TEST_CASE("SquareLossEqualTest", "[AdaBoostRegressorTest]")
{
  arma::rowvec values(10);
  values.fill(5);
  arma::rowvec loss = SquareLoss::Calculate(values);
  for(size_t i = 0; i < 10; i++)
    REQUIRE(loss[i] == Approx(1.0).margin(1e-5));
}

/**
 * Make sure the ExponentialLoss is of all elements is 0.6321 (1 - exp(-1)), 
 * when the elements are all equal.
 */
TEST_CASE("ExponentialLossEqualTest", "[AdaBoostRegressorTest]")
{
  arma::rowvec values(10);
  values.fill(5);
  arma::rowvec loss = ExponentialLoss::Calculate(values);
  for(size_t i = 0; i < 10; i++)
    REQUIRE(loss[i] == Approx(0.63212).margin(1e-5));
}

/**
 * Making sure the LinearLoss is evaluated correctly by doing calculation by
 * hand
 */
TEST_CASE("LinearLossHandCalculation", "[AdaBoostRegressorTest]")
{
  arma::rowvec values = {4., 2., 3., 4., 13., 6., 20., 8., 9., 10.};
  
  // Hand calculated loss.
  arma::rowvec loss = {0.2, 0.1, 0.15, 0.2, 0.65, 0.3, 1. , 0.4, 0.45, 0.5};

  arma::rowvec result = LinearLoss::Calculate(values);
  for(size_t i = 0; i < 10; i++)
    REQUIRE(result[i] == Approx(loss[i]).margin(1e-5));
}

/**
 * Making sure the SquareLoss is evaluated correctly by doing calculation by
 * hand
 */
TEST_CASE("SquareLossHandCalculation", "[AdaBoostRegressorTest]")
{
  arma::rowvec values = {4., 2., 3., 4., 13., 6., 20., 8., 9., 10.};
  
  // Hand calculated loss.
  arma::rowvec loss = {0.04, 0.01, 0.0225, 0.04, 0.4225, 0.09, 1., 0.16, 
                       0.2025, 0.25};

  arma::rowvec result = SquareLoss::Calculate(values);
  for(size_t i = 0; i < 10; i++)
    REQUIRE(result[i] == Approx(loss[i]).margin(1e-5));
}

/**
 * Making sure the ExponentialLoss is evaluated correctly by doing calculation by
 * hand
 */
TEST_CASE("ExponentialLossHandCalculation", "[AdaBoostRegressorTest]")
{
  arma::rowvec values = {4., 2., 3., 4., 13., 6., 20., 8., 9., 10.};
  
  // Hand calculated loss(rounded-off).
  arma::rowvec loss = {0.18126, 0.09516, 0.13929, 0.18126, 0.47795, 
                       0.25918, 0.63212, 0.32967, 0.36237, 0.39346};

  arma::rowvec result = ExponentialLoss::Calculate(values);
  for(size_t i = 0; i < 10; i++)
    REQUIRE(loss[i] == Approx(result[i]).margin(1e-5));
}

/**
 * Test that the model is able to perfectly fit all the obvious splits present
 * in the data and only use 1 tree.
 *
 *     |
 *     |
 *   2 |            xxxxxx
 *     |
 *     |
 *   1 |      xxxxxx      xxxxxx
 *     |
 *     |
 *   0 |xxxxxx                  xxxxxx
 *     |___________________________________
 */
TEST_CASE("EarlyTerminateTest", "[AdaBoostRegressorTest]")
{
  arma::mat dataset;
  arma::rowvec responses;
  arma::rowvec values = {0.0, 1.0, 2.0, 1.0, 0.0};

  CreateMultipleSplitData(dataset, responses, 1000, values);

  AdaBoostRegressor<> abr(dataset, responses, 20 /*numTrees*/, 1/*minLeaves*/, 
                        0/*minGainSplit*/, 0/*maxDepth*/);
  arma::rowvec preds;
  abr.Predict(dataset, preds);

  // Ensure that the predictions are perfect.
  for (size_t i = 0; i < responses.n_elem; ++i)
    REQUIRE(preds[i] == responses[i]);

  // Ensure that number of trees trained is 1, since 1st tree can do perfect 
  // predictions.
  REQUIRE(abr.NumTrees() == 1);
}

TEST_CASE("FloatEarlyTerminateTest", "[AdaBoostRegressorTest]")
{
  arma::fmat dataset;
  arma::frowvec responses;
  arma::frowvec values = {0.0, 1.0, 2.0, 1.0, 0.0};

  CreateMultipleSplitFloatData(dataset, responses, 1000, values);

  AdaBoostRegressor<> abr(dataset, responses, 20 /*numTrees*/, 1/*minLeaves*/, 
                        0/*minGainSplit*/, 0/*maxDepth*/);
  arma::frowvec preds;
  abr.Predict(dataset, preds);

  // Ensure that the predictions are perfect.
  for (size_t i = 0; i < responses.n_elem; ++i)
    REQUIRE(preds[i] == responses[i]);

  // Ensure that number of trees trained is 1, since 1st tree can do perfect 
  // predictions.
  REQUIRE(abr.NumTrees() == 1);
}

/**
 * Make sure an empty model cannot predict.
 */
TEST_CASE("EmptyPredictTest", "[AdaBoostRegressorTest]")
{
  AdaBoostRegressor<> abr; // No training.

  arma::mat points(10, 100, arma::fill::randu);
  arma::Row<double> predictions;
  REQUIRE_THROWS_AS(abr.Predict(points, predictions), std::invalid_argument);
  REQUIRE_THROWS_AS(abr.Predict(points.col(0)), std::invalid_argument);
}

TEST_CASE("FloatEmptyPredictTest", "[AdaBoostRegressorTest]")
{
  AdaBoostRegressor<> abr; // No training.

  arma::fmat points(10, 100, arma::fill::randu);
  arma::Row<float> predictions;
  REQUIRE_THROWS_AS(abr.Predict(points, predictions), std::invalid_argument);
  REQUIRE_THROWS_AS(abr.Predict(points.col(0)), std::invalid_argument);
}