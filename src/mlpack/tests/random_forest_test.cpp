/**
 * @file random_forest_test.cpp
 * @author Ryan Curtin
 *
 * Tests for the RandomForest class and related classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"

using namespace mlpack;
using namespace mlpack::tree;

BOOST_AUTO_TEST_SUITE(RandomForestTest);

/**
 * Make sure bootstrap sampling produces numbers in the dataset.
 */
BOOST_AUTO_TEST_CASE(BootstrapNoWeightsTest)
{
  arma::mat dataset(1, 1000);
  dataset.row(0) = arma::linspace<arma::rowvec>(1000, 1999, 1000);
  arma::Row<size_t> labels(1000);
  labels.fill(1); // Don't care about the labels.
  arma::rowvec weights; // Unused.

  // When we make bootstrap samples, they should include elements from 1k to 2k.
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat bootstrapDataset;
    arma::Row<size_t> bootstrapLabels;
    arma::rowvec bootstrapWeights;

    Bootstrap<false>(dataset, labels, weights, bootstrapDataset,
        bootstrapLabels, bootstrapWeights);

    BOOST_REQUIRE_EQUAL(bootstrapDataset.n_cols, 1000);
    BOOST_REQUIRE_EQUAL(bootstrapDataset.n_rows, 1);
    BOOST_REQUIRE_EQUAL(bootstrapLabels.n_elem, 1000);

    // Check each dataset element.
    for (size_t i = 0; i < dataset.n_cols; ++i)
    {
      BOOST_REQUIRE_GE(bootstrapDataset(0, i), 1000);
      BOOST_REQUIRE_LE(bootstrapDataset(0, i), 1999);
      BOOST_REQUIRE_EQUAL(bootstrapLabels[i], 1);
    }
  }
}

/**
 * Make sure bootstrap sampling produces numbers in the dataset.
 */
BOOST_AUTO_TEST_CASE(BootstrapWeightsTest)
{
  arma::mat dataset(1, 1000);
  dataset.row(0) = arma::linspace<arma::rowvec>(1000, 1999, 1000);
  arma::Row<size_t> labels(1000);
  labels.fill(1); // Don't care about the labels.
  arma::rowvec weights(1000, arma::fill::randu); // Unused.

  // When we make bootstrap samples, they should include elements from 1k to 2k.
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat bootstrapDataset;
    arma::Row<size_t> bootstrapLabels;
    arma::rowvec bootstrapWeights;

    Bootstrap<true>(dataset, labels, weights, bootstrapDataset,
        bootstrapLabels, bootstrapWeights);

    BOOST_REQUIRE_EQUAL(bootstrapDataset.n_cols, 1000);
    BOOST_REQUIRE_EQUAL(bootstrapDataset.n_rows, 1);
    BOOST_REQUIRE_EQUAL(bootstrapLabels.n_elem, 1000);
    BOOST_REQUIRE_EQUAL(bootstrapWeights.n_elem, 1000);

    // Check each dataset element.
    for (size_t i = 0; i < dataset.n_cols; ++i)
    {
      BOOST_REQUIRE_GE(bootstrapDataset(0, i), 1000);
      BOOST_REQUIRE_LE(bootstrapDataset(0, i), 1999);
      BOOST_REQUIRE_EQUAL(bootstrapLabels[i], 1);
      BOOST_REQUIRE_GE(bootstrapWeights[i], 0.0);
      BOOST_REQUIRE_LE(bootstrapWeights[i], 1.0);
    }
  }
}

/**
 * Make sure an empty forest cannot predict.
 */
BOOST_AUTO_TEST_CASE(EmptyClassifyTest)
{
  RandomForest<> rf; // No training.

  arma::mat points(10, 100, arma::fill::randu);
  arma::Row<size_t> predictions;
  arma::mat probabilities;
  size_t prediction;
  arma::vec pointProbabilities;
  BOOST_REQUIRE_THROW(rf.Classify(points, predictions), std::invalid_argument);
  BOOST_REQUIRE_THROW(rf.Classify(points.col(0)), std::invalid_argument);
  BOOST_REQUIRE_THROW(rf.Classify(points, predictions, probabilities),
      std::invalid_argument);
  BOOST_REQUIRE_THROW(rf.Classify(points.col(0), prediction,
      pointProbabilities), std::invalid_argument);
}

/**
 * Test unweighted numeric learning, making sure that we get better performance
 * than a single decision tree.
 */
BOOST_AUTO_TEST_CASE(UnweightedNumericLearningTest)
{
  // Load the vc2 dataset.
  arma::mat dataset;
  data::Load("vc2.csv", dataset);
  arma::Row<size_t> labels;
  data::Load("vc2_labels.txt", labels);

  // Build a random forest and a decision tree.
  RandomForest<GiniGain, RandomDimensionSelect> rf(dataset, labels, 3,
      10 /* 10 trees */, 5);
  DecisionTree<> dt(dataset, labels, 3, 5);

  // Get performance statistics on test data.
  arma::mat testDataset;
  data::Load("vc2_test.csv", testDataset);
  arma::Row<size_t> testLabels;
  data::Load("vc2_test_labels.txt", testLabels);

  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testDataset, rfPredictions);
  dt.Classify(testDataset, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  BOOST_REQUIRE_GE(rfCorrect, dtCorrect);
  BOOST_REQUIRE_GE(rfCorrect, size_t(0.7 * testDataset.n_cols));
}

/**
 * Test weighted numeric learning, making sure that we get better performance
 * than a single decision tree.
 */
BOOST_AUTO_TEST_CASE(WeightedNumericLearningTest)
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  data::Load("vc2.csv", dataset);
  data::Load("vc2_labels.txt", labels);

  // Add some noise.
  arma::mat noise(dataset.n_rows, 1000, arma::fill::randu);
  arma::Row<size_t> noiseLabels(1000);
  for (size_t i = 0; i < noiseLabels.n_elem; ++i)
    noiseLabels[i] = math::RandInt(3); // Random label.

  // Concatenate data matrices.
  arma::mat data = arma::join_rows(dataset, noise);
  arma::Row<size_t> fullLabels = arma::join_rows(labels, noiseLabels);

  // Now set weights.
  arma::rowvec weights(dataset.n_cols + 1000);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    weights[i] = math::Random(0.9, 1.0);
  for (size_t i = dataset.n_cols; i < dataset.n_cols + 1000; ++i)
    weights[i] = math::Random(0.0, 0.01); // Low weights for false points.

  // Train decision tree and random forest.
  RandomForest<GiniGain, RandomDimensionSelect> rf(dataset, labels, 3, weights,
      10, 5);
  DecisionTree<> dt(dataset, labels, 3, weights, 5);

  // Get performance statistics on test data.
  arma::mat testDataset;
  data::Load("vc2_test.csv", testDataset);
  arma::Row<size_t> testLabels;
  data::Load("vc2_test_labels.txt", testLabels);

  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testDataset, rfPredictions);
  dt.Classify(testDataset, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  BOOST_REQUIRE_GE(rfCorrect, dtCorrect);
  BOOST_REQUIRE_GE(rfCorrect, size_t(0.7 * testDataset.n_cols));
}

/**
 * Test unweighted categorical learning.  Ensure that we get better performance
 * with a random forest.
 */
BOOST_AUTO_TEST_CASE(UnweightedCategoricalLearningTest)
{
  arma::mat d;
  arma::Row<size_t> l;
  data::DatasetInfo di;
  MockCategoricalData(d, l, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::Row<size_t> trainingLabels = l.subvec(0, 1999);
  arma::Row<size_t> testLabels = l.subvec(2000, 3999);

  // Train a random forest and a decision tree.
  RandomForest<> rf(trainingData, di, trainingLabels, 5, 15 /* 15 trees */, 5);
  DecisionTree<> dt(trainingData, di, trainingLabels, 5, 5);

  // Get performance statistics on test data.
  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testData, rfPredictions);
  dt.Classify(testData, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  BOOST_REQUIRE_GE(rfCorrect, dtCorrect - 30);
  BOOST_REQUIRE_GE(rfCorrect, size_t(0.7 * testData.n_cols));
}

/**
 * Test weighted categorical learning.
 */
BOOST_AUTO_TEST_CASE(WeightedCategoricalLearningTest)
{
  arma::mat d;
  arma::Row<size_t> l;
  data::DatasetInfo di;
  MockCategoricalData(d, l, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::Row<size_t> trainingLabels = l.subvec(0, 1999);
  arma::Row<size_t> testLabels = l.subvec(2000, 3999);

  // Now create random points.
  arma::mat randomNoise(4, 2000);
  arma::Row<size_t> randomLabels(2000);
  for (size_t i = 0; i < 2000; ++i)
  {
    randomNoise(0, i) = math::Random();
    randomNoise(1, i) = math::Random();
    randomNoise(2, i) = math::RandInt(4);
    randomNoise(3, i) = math::RandInt(2);
    randomLabels[i] = math::RandInt(5);
  }

  // Generate weights.
  arma::rowvec weights(4000);
  for (size_t i = 0; i < 2000; ++i)
    weights[i] = math::Random(0.9, 1.0);
  for (size_t i = 2000; i < 4000; ++i)
    weights[i] = math::Random(0.0, 0.001);

  arma::mat fullData = arma::join_rows(trainingData, randomNoise);
  arma::Row<size_t> fullLabels = arma::join_rows(trainingLabels, randomLabels);

  // Build a random forest and a decision tree.
  RandomForest<> rf(fullData, di, fullLabels, 5, 15 /* 15 trees */, 5);
  DecisionTree<> dt(fullData, di, fullLabels, 5, 5);

  // Get performance statistics on test data.
  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testData, rfPredictions);
  dt.Classify(testData, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  BOOST_REQUIRE_GE(rfCorrect, dtCorrect - 30);
  BOOST_REQUIRE_GE(rfCorrect, size_t(0.7 * testData.n_cols));
}

/**
 * Test that learning with a leaf size of 1 successfully memorizes the training
 * set.
 */
BOOST_AUTO_TEST_CASE(LeafSize1Test)
{
  // Load the vc2 dataset.
  arma::mat dataset;
  data::Load("vc2.csv", dataset);
  arma::Row<size_t> labels;
  data::Load("vc2_labels.txt", labels);

  // Build a random forest with a leaf size of 1.
  RandomForest<> rf(dataset, labels, 3, 10 /* 10 trees */, 1);

  // Predict on the training set.
  arma::Row<size_t> predictions;
  rf.Classify(dataset, predictions);

  const size_t correct = arma::accu(predictions == labels);
  BOOST_REQUIRE_EQUAL(correct, dataset.n_cols);
}

/**
 * Test that a leaf size equal to the dataset size learns nothing.
 */
BOOST_AUTO_TEST_CASE(LeafSizeDatasetTest)
{
  // Load the vc2 dataset.
  arma::mat dataset;
  data::Load("vc2.csv", dataset);
  arma::Row<size_t> labels;
  data::Load("vc2_labels.txt", labels);

  // Build a random forest with a leaf size equal to the number of points in the
  // dataset.
  RandomForest<> rf(dataset, labels, 3, 10 /* 10 trees */, dataset.n_cols);

  // Calculate majority probabilities.
  arma::vec majorityProbs(3, arma::fill::zeros);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    majorityProbs[labels[i]]++;
  majorityProbs /= dataset.n_cols;
  arma::uword max;
  majorityProbs.max(max);
  size_t majorityClass = (size_t) max;

  // Predict on the training set.
  arma::Row<size_t> predictions;
  arma::mat probabilities;
  rf.Classify(dataset, predictions, probabilities);

  BOOST_REQUIRE_EQUAL(probabilities.n_rows, 3);
  BOOST_REQUIRE_EQUAL(probabilities.n_cols, dataset.n_cols);
  BOOST_REQUIRE_EQUAL(predictions.n_elem, dataset.n_cols);
  for (size_t i = 0; i < predictions.n_cols; ++i)
  {
    BOOST_REQUIRE_EQUAL(predictions[i], majorityClass);
    for (size_t j = 0; j < probabilities.n_rows; ++j)
      BOOST_REQUIRE_CLOSE(probabilities(j, i), majorityProbs[j], 1e-5);
  }
}

// Make sure we can serialize a random forest.
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  // Load the vc2 dataset.
  arma::mat dataset;
  data::Load("vc2.csv", dataset);
  arma::Row<size_t> labels;
  data::Load("vc2_labels.txt", labels);

  RandomForest<> rf(dataset, labels, 3, 10 /* 10 trees */, 10);

  arma::Row<size_t> beforePredictions;
  arma::mat beforeProbabilities;
  rf.Classify(dataset, beforePredictions, beforeProbabilities);

  RandomForest<> xmlForest, textForest, binaryForest;
  binaryForest.Train(dataset, labels, 3, 3, 50);
  SerializeObjectAll(rf, xmlForest, textForest, binaryForest);

  // Now check that we get the same results serializing other things.
  arma::Row<size_t> xmlPredictions, textPredictions, binaryPredictions;
  arma::mat xmlProbabilities, textProbabilities, binaryProbabilities;

  xmlForest.Classify(dataset, xmlPredictions, xmlProbabilities);
  textForest.Classify(dataset, textPredictions, textProbabilities);
  binaryForest.Classify(dataset, binaryPredictions, binaryProbabilities);

  CheckMatrices(beforePredictions, xmlPredictions, textPredictions,
      binaryPredictions);
  CheckMatrices(beforeProbabilities, xmlProbabilities, textProbabilities,
      binaryProbabilities);
}

BOOST_AUTO_TEST_SUITE_END();
