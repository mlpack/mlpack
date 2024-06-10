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
 * This test case runs the GradBoosting model on the UCI Iris dataset. 
 * Tests if the model gives a training accuracy > 60, without 
 * initiating a weak learner. Used default empty constructor.
*/
TEST_CASE("GBTrainMethod1", "[GradBoostGeneralTest]") 
{
  arma::mat db;
  if (!data::Load("iris.csv", db))
    FAIL("Cannot load test dataset iris.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;

  GradBoosting gb;

  gb.Train(db, labels, numClasses, numModels);

  arma::Row<size_t> predictions;
  gb.Classify(db, predictions);

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
 * This test case runs the GradBoosting model on the UCI Iris dataset.
 * Tests if the model gives a training accuracy > 60 given the data and a 
 * pre-initiated weak learner. Used default empty constructor.
*/
TEST_CASE("GBTrainMethod2", "[GradBoostGeneralTest]") 
{
  arma::mat db;
  if (!data::Load("iris.csv", db))
    FAIL("Cannot load test dataset iris.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = arma::max(labels.row(0)) + 1;
  const size_t numModels = 5;
  mlpack::DecisionTree<GiniGain,
                        BestBinaryNumericSplit,
                        AllCategoricalSplit,
                        AllDimensionSelect,
                        false> tree;

  GradBoosting gb;

  gb.Train(db, labels, numClasses, numModels, tree);

  arma::Row<size_t> predictions;
  gb.Classify(db, predictions);

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
 * This test case runs the GradBoosting model on the UCI Iris dataset.
 * Tests if the model gives a training accuracy > 60 given the data and a 
 * weak learner arguments. Used default empty constructor.
*/
TEST_CASE("GBTrainMethod3", "[GradBoostGeneralTest]") 
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

  GradBoosting gb;

  gb.Train(db, 
            labels, 
            numClasses, 
            numModels, 
            minimumLeafSize,
            minimumGainSplit,
            maximumDepth,
            dimensionSelector);

  arma::Row<size_t> predictions;
  gb.Classify(db, predictions);

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