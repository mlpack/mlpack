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
#include <mlpack/methods/xgboost.hpp>

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
 * This test case runs the XGBoost model on the Iris dataset. 
 * Tests if the model gives a training accuracy > 60, without 
 * initiating a weak learner. Used default empty constructor.
*/
TEST_CASE("XGBIrisTrainMethod1", "[XGBoostGeneralTest]") 
{
  arma::mat db;
  if (!data::Load("iris.csv", db))
    FAIL("Cannot load test dataset iris.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb;

  xgb.Train(db, labels, numClasses, numModels);

  arma::Row<size_t> predictions;
  xgb.Classify(db, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++) 
  {
    if(labels(i) == predictions(i)) 
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);

}

/**
 * This test case runs the XGBoost model on the Iris dataset.
 * Tests if the model gives a training accuracy > 60 given the data and a 
 * weak learner arguments. Used default empty constructor.
*/
TEST_CASE("XGBIrisTrainMethod2", "[XGBoostGeneralTest]") 
{
  arma::mat db;
  if (!data::Load("iris.csv", db))
    FAIL("Cannot load test dataset iris.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  const size_t minimumLeafSize=10;
  const double minimumGainSplit=1e-7;
  const size_t maximumDepth=2;
  const AllDimensionSelect dimensionSelector;

  XGBoost xgb;

  xgb.Train(db, 
            labels, 
            numClasses, 
            numModels, 
            minimumLeafSize,
            minimumGainSplit,
            maximumDepth);

  arma::Row<size_t> predictions;
  xgb.Classify(db, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++) 
  {
    if(labels(i) == predictions(i)) 
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);

}

/**
 * This test case runs the XGBoost model on the Iris dataset.
 * Tests if the model gives a testing accuracy > 60 given the data.
 * Used default empty constructor.
*/
TEST_CASE("XGBIrisTestAccuracy", "[XGBoostGeneralTest]") 
{
  arma::mat db;
  if (!data::Load("iris_train.csv", db))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  arma::mat testDb;
  if (!data::Load("iris_test.csv", testDb))
    FAIL("Cannot load test dataset iris_test.csv!");

  arma::Row<size_t> testLabels;
  if (!data::Load("iris_test_labels.csv", testLabels))
    FAIL("Cannot load test dataset iris_test_labels.csv!");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb;

  xgb.Train(db, labels, numClasses, numModels);

  arma::Row<size_t> predictions;
  xgb.Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++) 
  {
    if(testLabels(i) == predictions(i)) 
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);

}

/**
 * Testing the Weak Learner fetching function
 */
TEST_CASE("XGBWeakLearnerFunction", "[XGBoostUnitTest]")
{
  arma::mat db;
  if (!data::Load("iris_train.csv", db))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  arma::mat testDb;
  if (!data::Load("iris_test.csv", testDb))
    FAIL("Cannot load test dataset iris_test.csv!");

  arma::Row<size_t> testLabels;
  if (!data::Load("iris_test_labels.csv", testLabels))
    FAIL("Cannot load test dataset iris_test_labels.csv!");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb;
  
  xgb.Train(db, labels, numClasses, numModels);

  arma::Row<size_t> predictions;
  xgb.WeakLearner(0).Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++) 
  {
    if(testLabels(i) == predictions(i)) 
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 0);

}