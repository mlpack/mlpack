/**
 * @file tests/grad_boosting_test.cpp
 * @author Abhimanyu Dayal
 *
 * Tests for GradBoosting class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/grad_boosting.hpp>

#include "serialization.hpp"
#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace data;

/**
 * This test case runs the GradBoosting model on the Iris dataset. 
 * Tests if the model gives a training accuracy > 60, without 
 * initiating a weak learner. Used default empty constructor.
*/
TEST_CASE("GBIrisTrainMethod1", "[GradBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris.csv", db, info))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numWeakLearners = 5;

  GradBoosting gb;

  gb.Train(db, info, labels, numClasses, numWeakLearners);

  arma::Row<size_t> predictions;
  gb.Classify(db, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++)
  {
    if (labels(i) == predictions(i))
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);
}

/**
 * This test case runs the GradBoosting model on the Iris dataset.
 * Tests if the model gives a training accuracy > 60 given the data and a 
 * weak learner arguments. Used default empty constructor.
*/
TEST_CASE("GBIrisTrainMethod2", "[GradBoostGeneralTest]")
{
  arma::mat db;
  mlpack::data::DatasetInfo info;
  if (!data::Load("iris.csv", db, info))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numWeakLearners = 5;

  const size_t minimumLeafSize = 10;
  const double minimumGainSplit = 1e-7;
  const size_t maximumDepth = 2;

  GradBoosting gb;

  gb.Train(db, info, labels, numClasses, numWeakLearners,
            minimumLeafSize, minimumGainSplit, maximumDepth);

  arma::Row<size_t> predictions;
  gb.Classify(db, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < labels.n_elem; i++)
  {
    if (labels(i) == predictions(i))
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) labels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);
}

/**
 * This test case runs the GradBoosting model on the Iris dataset.
 * Tests if the model gives a testing accuracy > 60 given the data.
 * Used default empty constructor.
*/
TEST_CASE("GBIrisTestAccuracy", "[GradBoostGeneralTest]")
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
  const size_t numWeakLearners = 5;

  GradBoosting gb;

  gb.Train(db, info, labels, numClasses, numWeakLearners);

  arma::Row<size_t> predictions;
  gb.Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++)
  {
    if (testLabels(i) == predictions(i))
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);
}

/**
 * Check the second constructor.
*/
TEST_CASE("GBConstr1", "[GradBoostGeneralTest]")
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
  const size_t numWeakLearners = 5;

  GradBoosting gb(db, info, labels, numClasses, numWeakLearners);

  arma::Row<size_t> predictions;
  gb.Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++)
  {
    if (testLabels(i) == predictions(i))
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);
}

/**
 * Check the third constructor.
*/
TEST_CASE("GBConstr2", "[GradBoostGeneralTest]")
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
  const size_t numWeakLearners = 5;
  const size_t minimumLeafSize = 10;
  const double minimumGainSplit = 1e-7;
  const size_t maximumDepth = 2;

  GradBoosting gb(db, info, labels, numClasses, numWeakLearners,
    minimumLeafSize, minimumGainSplit, maximumDepth);

  arma::Row<size_t> predictions;
  gb.Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++)
  {
    if (testLabels(i) == predictions(i))
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);
}


/**
 * Check the first classify method.
*/
TEST_CASE("GBClassify1", "[GradBoostGeneralTest]")
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
  const size_t numWeakLearners = 5;

  GradBoosting gb(db, info, labels, numClasses, numWeakLearners);

  size_t prediction = gb.Classify(testDb.col(0));

  REQUIRE(prediction == testLabels(0));
}

/**
 * Check the second classify method.
*/
TEST_CASE("GBClassify2", "[GradBoostGeneralTest]")
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
  const size_t numWeakLearners = 5;

  GradBoosting gb(db, info, labels, numClasses, numWeakLearners);

  size_t prediction;
  gb.Classify(testDb.col(0), prediction);

  REQUIRE(prediction == testLabels(0));
}

/**
 * Check the third classify method.
*/
TEST_CASE("GBClassify3", "[GradBoostGeneralTest]")
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
  const size_t numWeakLearners = 5;

  GradBoosting gb(db, info, labels, numClasses, numWeakLearners);

  arma::Row<size_t> predictions;
  gb.Classify(testDb, predictions);

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++)
  {
    if (testLabels(i) == predictions(i))
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 90);
}
