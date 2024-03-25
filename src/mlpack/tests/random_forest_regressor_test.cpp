/**
 * @file tests/decision_tree_regressor_test.cpp
 * @author Dinesh Kumar
 *
 * Tests for the DecisionTreeRegressor class and related classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"
#include "test_function_tools.hpp"

using namespace mlpack;

/**
 * Make sure an empty forest cannot predict.
 */
TEST_CASE("EmptyPredictTest", "[RandomForestRegressorTest]")
{
  RandomForestRegressor<> rf; // No training.

  arma::mat points(10, 100, arma::fill::randu);
  arma::Row<double> predictions;
  REQUIRE_THROWS_AS(rf.Predict(points, predictions), std::invalid_argument);
  REQUIRE_THROWS_AS(rf.Predict(points.col(0)), std::invalid_argument);
}

/**
 * Test unweighted numeric learning, making sure that we get better performance
 * than a single decision tree.
 */
TEST_CASE("UnweightedLearnTest", "[RandomForestRegressorTest]")
{
  // Loading data.
  data::DatasetInfo info;
  arma::mat trainData, testData;
  arma::rowvec trainResponses, testResponses;
  LoadBostonHousingDataset(trainData, testData, trainResponses, testResponses, info);

  mlpack::RandomForestRegressor<> rfr;
  mlpack::DecisionTreeRegressor<> dtr;

  rfr.Train(trainData, trainResponses);
  dtr.Train(trainData, trainResponses);
  
  double testMean = arma::mean(testResponses);

  arma::Row<double> rfrPred;
  rfr.Predict(testData, rfrPred);
  // Calculate r2 score for random forest regressor.
  double rfrR2Score = 1 - arma::sum(square(testResponses - rfrPred)) / arma::sum(square(testResponses - testMean));

  arma::Row<double> dtrPred;
  dtr.Predict(testData, dtrPred);
  // Calculate r2 score for decision tree regressor.
  double dtrR2Score = 1 - arma::sum(square(testResponses - dtrPred)) / arma::sum(square(testResponses - testMean));

  REQUIRE(rfrR2Score > dtrR2Score);
}

TEST_CASE("WarmStartRegressorTreesTest", "[RandomForestRegressorTest]")
{
  // Loading data.
  data::DatasetInfo info;
  arma::mat trainData, testData;
  arma::rowvec trainResponses, testResponses;
  LoadBostonHousingDataset(trainData, testData, trainResponses, testResponses, info);

  // Train a random forest.
  RandomForestRegressor<> rfr(trainData, info, trainResponses, 25 /* 25 trees */, 1,
      1e-7, 0, MultipleRandomDimensionSelect(4));

  REQUIRE(rfr.NumTrees() == 25);

  rfr.Train(trainData, info, trainResponses, 20 /* 20 trees */, 1, 1e-7, 0,
      true /* warmStart */, MultipleRandomDimensionSelect(4));

  REQUIRE(rfr.NumTrees() == 25 + 20);
}
