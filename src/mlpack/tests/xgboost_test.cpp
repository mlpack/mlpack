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
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris.csv", db, info))
    FAIL("Cannot load test dataset iris.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb;

  xgb.Train(db, labels, info, numClasses, numModels);

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
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris.csv", db, info))
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

  xgb.Train(db, labels, info,numClasses, numModels, 
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
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
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

  xgb.Train(db, labels, info, numClasses, numModels);

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
 * Check if the Pruning method is removing nodes if they 
 * don't meet threshold conditions.
 */
TEST_CASE("PruningBaseTest", "[XGBoostGeneralTest]")
{
  arma::mat dataset;
  mlpack::data::DatasetInfo info;
  if (!data::Load("vc2.csv", dataset, info))
    FAIL("Cannot load test dataset vc2.csv!");
  
  arma::rowvec labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt!");
  
  FeatureImportance* featImp = new FeatureImportance();

  const arma::mat db = dataset;

  XGBTree d(db, labels, info, 10, 1e-7, 3, featImp);

  // Set the threshold high to ensure that it deletes root node.
  bool flag = d.Prune(10);

  REQUIRE(flag == true);
}

/**
 * Confirm the XGBoost Test accuracy is as required.
 */
TEST_CASE("XGBoostTestAccuracy", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
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

  xgb.Train(db, labels, info, numClasses, numModels);

  arma::Row<size_t> predictions;
  xgb.Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++) 
    if(testLabels(i) == predictions(i)) 
      accuracy++;

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 90);
}

/**
 * Check the 2nd XGBoost constructor.
 */
TEST_CASE("XGBoostConstr2", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb(db, labels, info, numClasses, numModels);
  
  arma::Row<size_t> predictions;
  xgb.Classify(db, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++) 
    if(labels(i) == predictions(i)) 
      accuracy++;

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 90);

}


/**
 * Check the 3rd XGBoost constructor.
 */
TEST_CASE("XGBoostConstr3", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;
  const size_t minimumLeafSize = 10;
  const double minimumGainSplit = 1e-7;
  const size_t maximumDepth = 2;

  XGBoost xgb(db, labels, info, numClasses, numModels, 
    minimumLeafSize, minimumGainSplit, maximumDepth);
  
  arma::Row<size_t> predictions;
  xgb.Classify(db, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++) 
    if(labels(i) == predictions(i)) 
      accuracy++;

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 90);

}

/**
 * Check the 1st Classify method.
 */
TEST_CASE("XGBoostClassify1", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb(db, labels, info, numClasses, numModels);

  size_t label = xgb.Classify(db.col(0));

  REQUIRE(label == labels(0));

}

/**
 * Check the 2nd Classify method.
 */
TEST_CASE("XGBoostClassify2", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb(db, labels, info, numClasses, numModels);

  size_t label;
  arma::rowvec probabilities;

  xgb.Classify(db.col(0), label, probabilities);

  REQUIRE(label == labels(0));

}


/**
 * Check the 3rd Classify method.
 */
TEST_CASE("XGBoostClassify3", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  XGBoost xgb(db, labels, info, numClasses, numModels);

  size_t label;

  xgb.Classify(db.col(0), label);

  REQUIRE(label == labels(0));

}

/**
 * Check the 4th Classify method.
 */
TEST_CASE("XGBoostClassify4", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
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

  XGBoost xgb(db, labels, info, numClasses, numModels);

  arma::Row<size_t> predictions;
  arma::mat probabilities;

  xgb.Classify(testDb, predictions, probabilities);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++) 
    if(labels(i) == predictions(i)) 
      accuracy++;

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 90);
}


/**
 * Check the 5th Classify method.
 */
TEST_CASE("XGBoostClassify5", "[XGBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris_train.csv", db, info))
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

  XGBoost xgb(db, labels, info, numClasses, numModels);

  arma::Row<size_t> predictions;

  xgb.Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++) 
    if(labels(i) == predictions(i)) 
      accuracy++;

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 90);
}


