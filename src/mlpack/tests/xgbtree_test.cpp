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
#include <mlpack/methods/xgboost/xgbtree.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"

using namespace mlpack;

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
