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
 * Tests if the model gives a valid output given only the data, without 
 * initiating a weak learner.
*/
TEST_CASE("GBUninitiatedTest", "[GradBoostGeneralTest]") 
{
  arma::mat db;
  if (!data::Load("iris.csv", db))
    FAIL("Cannot load test dataset iris.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;
  const size_t numModels = 10;

  GradBoosting gb;

  gb.Train(db, labels, numClasses, numModels);

  arma::mat testDb(db.n_rows, db.n_cols, arma::fill::randu);
  arma::Row<size_t> predictions;

  gb.Classify(testDb, predictions);

  predictions.print();

}

/**
 * This test case runs the GradBoosting model on the UCI Iris dataset.
 * Tests if the model gives a valid output given the data and a pre-initiated 
 * weak learner.
*/
TEST_CASE("GBInitiatedTest", "[GradBoostGeneralTest]") 
{
  arma::mat db;
  if (!data::Load("iris.csv", db))
    FAIL("Cannot load test dataset iris.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;
  const size_t numModels = 10;
  ID3DecisionStump tree;

  GradBoosting gb;

  gb.Train(db, labels, numClasses, numModels, tree);

  arma::mat testDb(db.n_rows, db.n_cols, arma::fill::randu);
  arma::Row<size_t> predictions;

  gb.Classify(testDb, predictions);

  predictions.print();

}