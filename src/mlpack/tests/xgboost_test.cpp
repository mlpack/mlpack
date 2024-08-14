/**
 * @file tests/xgboost_test.cpp
 * @author Abhimanyu Dayal
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
#include <mlpack/methods/xgboost/xgbtree.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;

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

/**
 * Check if the Pruning method is removing nodes if they 
 * don't meet threshold conditions.
 */

TEST_CASE("PruningBaseTest", "[XGBTreeTest]")
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load test dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt!");

  XGBTree d(dataset, labels, 3, 10, 1e-7, 1);

  // Set the threshold high to ensure that it deletes root node.
  bool flag = d.Prune(10);

  REQUIRE(flag == true);
}

/*
 * Test that we can pass const data into XGBTree constructors.
 */
TEST_CASE("ConstDataTest", "[XGBTreeTest]")
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);

  const arma::mat& constData = data;
  const arma::Row<size_t>& constLabels = labels;
  const arma::rowvec constWeights(labels.n_elem, arma::fill::randu);
  const size_t numClasses = 5;

  XGBTree<> dt(constData, constLabels, numClasses);
  XGBTree<> dt2(constData, datasetInfo, constLabels, numClasses);
  XGBTree<> dt3(constData, constLabels, numClasses, constWeights);
  XGBTree<> dt4(constData, datasetInfo, constLabels, numClasses,
      constWeights);
}
