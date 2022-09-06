/**
 * @file tests/random_forest_test.cpp
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
#include <mlpack/methods/random_forest.hpp>

#include "serialization.hpp"
#include "test_catch_tools.hpp"
#include "catch.hpp"
#include "mock_categorical_data.hpp"

using namespace mlpack;

/**
 * Make sure bootstrap sampling produces numbers in the dataset.
 */
TEST_CASE("BootstrapNoWeightsTest", "[RandomForestTest]")
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

    REQUIRE(bootstrapDataset.n_cols == 1000);
    REQUIRE(bootstrapDataset.n_rows == 1);
    REQUIRE(bootstrapLabels.n_elem == 1000);

    // Check each dataset element.
    for (size_t i = 0; i < dataset.n_cols; ++i)
    {
      REQUIRE(bootstrapDataset(0, i) >= 1000);
      REQUIRE(bootstrapDataset(0, i) <= 1999);
      REQUIRE(bootstrapLabels[i] == 1);
    }
  }
}

/**
 * Make sure bootstrap sampling produces numbers in the dataset.
 */
TEST_CASE("BootstrapWeightsTest", "[RandomForestTest]")
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

    REQUIRE(bootstrapDataset.n_cols == 1000);
    REQUIRE(bootstrapDataset.n_rows == 1);
    REQUIRE(bootstrapLabels.n_elem == 1000);
    REQUIRE(bootstrapWeights.n_elem == 1000);

    // Check each dataset element.
    for (size_t i = 0; i < dataset.n_cols; ++i)
    {
      REQUIRE(bootstrapDataset(0, i) >= 1000);
      REQUIRE(bootstrapDataset(0, i) <= 1999);
      REQUIRE(bootstrapLabels[i] == 1);
      REQUIRE(bootstrapWeights[i] >= 0.0);
      REQUIRE(bootstrapWeights[i] <= 1.0);
    }
  }
}

/**
 * Make sure an empty forest cannot predict.
 */
TEST_CASE("EmptyClassifyTest", "[RandomForestTest]")
{
  RandomForest<> rf; // No training.

  arma::mat points(10, 100, arma::fill::randu);
  arma::Row<size_t> predictions;
  arma::mat probabilities;
  size_t prediction;
  arma::vec pointProbabilities;
  REQUIRE_THROWS_AS(rf.Classify(points, predictions), std::invalid_argument);
  REQUIRE_THROWS_AS(rf.Classify(points.col(0)), std::invalid_argument);
  REQUIRE_THROWS_AS(rf.Classify(points, predictions, probabilities),
      std::invalid_argument);
  REQUIRE_THROWS_AS(rf.Classify(points.col(0), prediction,
      pointProbabilities), std::invalid_argument);
}

/**
 * Test unweighted numeric learning, making sure that we get better performance
 * than a single decision tree.
 */
TEST_CASE("UnweightedNumericLearningTest", "[RandomForestTest]")
{
  // Load the vc2 dataset.
  arma::mat dataset;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load dataset vc2.csv");
  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load dataset vc2.csv");

  // Build a random forest and a decision tree.
  RandomForest<> rf(dataset, labels, 3, 20 /* 20 trees */, 1, 1e-7);
  DecisionTree<> dt(dataset, labels, 3, 5);

  // Get performance statistics on test data.
  arma::mat testDataset;
  if (!data::Load("vc2_test.csv", testDataset))
    FAIL("Cannot load dataset vc2_test.csv");
  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    FAIL("Cannot load dataset vc2_test_labels.txt");

  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testDataset, rfPredictions);
  dt.Classify(testDataset, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  REQUIRE(rfCorrect >= dtCorrect * 0.9);
  REQUIRE(rfCorrect >= size_t(0.7 * testDataset.n_cols));
}

/**
 * Test weighted numeric learning, making sure that we get better performance
 * than a single decision tree.
 */
TEST_CASE("WeightedNumericLearningTest", "[RandomForestTest]")
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load dataset vc2.csv");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load dataset vc2_labels.txt");

  // Add some noise.
  arma::mat noise(dataset.n_rows, 1000, arma::fill::randu);
  arma::Row<size_t> noiseLabels(1000);
  for (size_t i = 0; i < noiseLabels.n_elem; ++i)
    noiseLabels[i] = RandInt(3); // Random label.

  // Concatenate data matrices.
  arma::mat data = arma::join_rows(dataset, noise);
  arma::Row<size_t> fullLabels = arma::join_rows(labels, noiseLabels);

  // Now set weights.
  arma::rowvec weights(dataset.n_cols + 1000);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = dataset.n_cols; i < dataset.n_cols + 1000; ++i)
    weights[i] = Random(0.0, 0.01); // Low weights for false points.

  // Train decision tree and random forest.
  RandomForest<> rf(dataset, labels, 3, weights, 20, 1);
  DecisionTree<> dt(dataset, labels, 3, weights, 5);

  // Get performance statistics on test data.
  arma::mat testDataset;
  if (!data::Load("vc2_test.csv", testDataset))
    FAIL("Cannot load dataset vc2_test.csv");
  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    FAIL("Cannot load dataset vc2_test_labels.txt");

  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testDataset, rfPredictions);
  dt.Classify(testDataset, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  REQUIRE(rfCorrect >= dtCorrect * 0.9);
  REQUIRE(rfCorrect >= size_t(0.7 * testDataset.n_cols));
}

/**
 * Test unweighted categorical learning.  Ensure that we get better performance
 * with a random forest.
 */
TEST_CASE("UnweightedCategoricalLearningTest", "[RandomForestTest]")
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
  RandomForest<> rf(trainingData, di, trainingLabels, 5, 25 /* 25 trees */, 1,
      1e-7, 0, MultipleRandomDimensionSelect(4));
  DecisionTree<> dt(trainingData, di, trainingLabels, 5, 5);

  // Get performance statistics on test data.
  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testData, rfPredictions);
  dt.Classify(testData, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  REQUIRE(rfCorrect >= dtCorrect - 25);
  REQUIRE(rfCorrect >= size_t(0.7 * testData.n_cols));
}

/**
 * Test weighted categorical learning.
 */
TEST_CASE("WeightedCategoricalLearningTest", "[RandomForestTest]")
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
    randomNoise(0, i) = Random();
    randomNoise(1, i) = Random();
    randomNoise(2, i) = RandInt(4);
    randomNoise(3, i) = RandInt(2);
    randomLabels[i] = RandInt(5);
  }

  // Generate weights.
  arma::rowvec weights(4000);
  for (size_t i = 0; i < 2000; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = 2000; i < 4000; ++i)
    weights[i] = Random(0.0, 0.001);

  arma::mat fullData = arma::join_rows(trainingData, randomNoise);
  arma::Row<size_t> fullLabels = arma::join_rows(trainingLabels, randomLabels);

  // Build a random forest and a decision tree.
  RandomForest<> rf(fullData, di, fullLabels, 5, weights, 25 /* 25 trees */, 1,
      1e-7, 0, MultipleRandomDimensionSelect(4));
  DecisionTree<> dt(fullData, di, fullLabels, 5, weights, 5);

  // Get performance statistics on test data.
  arma::Row<size_t> rfPredictions;
  arma::Row<size_t> dtPredictions;

  rf.Classify(testData, rfPredictions);
  dt.Classify(testData, dtPredictions);

  // Calculate the number of correct points.
  size_t rfCorrect = arma::accu(rfPredictions == testLabels);
  size_t dtCorrect = arma::accu(dtPredictions == testLabels);

  REQUIRE(rfCorrect >= dtCorrect - 25);
  REQUIRE(rfCorrect >= size_t(0.7 * testData.n_cols));
}

/**
 * Test that a leaf size equal to the dataset size learns nothing.
 */
TEST_CASE("LeafSizeDatasetTest", "[RandomForestTest]")
{
  // Load the vc2 dataset.
  arma::mat dataset;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load dataset vc2.csv");
  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load dataset vc2.csv");

  // Build a random forest with a leaf size equal to the number of points in the
  // dataset.
  RandomForest<> rf(dataset, labels, 3, 10 /* 10 trees */, dataset.n_cols);

  // Predict on the training set.
  arma::Row<size_t> predictions;
  arma::mat probabilities;
  rf.Classify(dataset, predictions, probabilities);

  // We want to check that all the classes and probabilities are the same for
  // all predictions.
  size_t majorityClass = predictions[0];
  arma::vec majorityProbs = probabilities.col(0);

  REQUIRE(probabilities.n_rows == 3);
  REQUIRE(probabilities.n_cols == dataset.n_cols);
  REQUIRE(predictions.n_elem == dataset.n_cols);
  for (size_t i = 1; i < predictions.n_cols; ++i)
  {
    REQUIRE(predictions[i] == majorityClass);
    for (size_t j = 0; j < probabilities.n_rows; ++j)
      REQUIRE(probabilities(j, i) == Approx(majorityProbs[j]).epsilon(1e-7));
  }
}

// Make sure we can serialize a random forest.
TEST_CASE("RandomForestSerializationTest", "[RandomForestTest]")
{
  // Load the vc2 dataset.
  arma::mat dataset;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load dataset vc2.csv");
  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load dataset vc2.csv");

  RandomForest<> rf(dataset, labels, 3, 10 /* 10 trees */, 1);

  arma::Row<size_t> beforePredictions;
  arma::mat beforeProbabilities;
  rf.Classify(dataset, beforePredictions, beforeProbabilities);

  RandomForest<> xmlForest, jsonForest, binaryForest;
  binaryForest.Train(dataset, labels, 3, 3, 5, 1);
  SerializeObjectAll(rf, xmlForest, jsonForest, binaryForest);

  // Now check that we get the same results serializing other things.
  arma::Row<size_t> xmlPredictions, jsonPredictions, binaryPredictions;
  arma::mat xmlProbabilities, jsonProbabilities, binaryProbabilities;

  xmlForest.Classify(dataset, xmlPredictions, xmlProbabilities);
  jsonForest.Classify(dataset, jsonPredictions, jsonProbabilities);
  binaryForest.Classify(dataset, binaryPredictions, binaryProbabilities);

  CheckMatrices(beforePredictions, xmlPredictions, jsonPredictions,
      binaryPredictions);
  CheckMatrices(beforeProbabilities, xmlProbabilities, jsonProbabilities,
      binaryProbabilities);
}

/**
 * Test that RandomForest::Train() returns finite average entropy on numeric
 * dataset.
 */
TEST_CASE("RandomForestNumericTrainReturnEntropy", "[RandomForestTest]")
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load dataset vc2.csv");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load dataset vc2_labels.txt");

  // Add some noise.
  arma::mat noise(dataset.n_rows, 1000, arma::fill::randu);
  arma::Row<size_t> noiseLabels(1000);
  for (size_t i = 0; i < noiseLabels.n_elem; ++i)
    noiseLabels[i] = RandInt(3); // Random label.

  // Concatenate data matrices.
  arma::mat data = arma::join_rows(dataset, noise);
  arma::Row<size_t> fullLabels = arma::join_rows(labels, noiseLabels);

  // Now set weights.
  arma::rowvec weights(dataset.n_cols + 1000);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = dataset.n_cols; i < dataset.n_cols + 1000; ++i)
    weights[i] = Random(0.0, 0.01); // Low weights for false points.

  // Test random forest on unweighted numeric dataset.
  RandomForest<GiniGain, RandomDimensionSelect> rf;
  double entropy = rf.Train(dataset, labels, 3, 10, 1);

  REQUIRE(std::isfinite(entropy) == true);

  // Test random forest on weighted numeric dataset.
  RandomForest<GiniGain, RandomDimensionSelect> wrf;
  entropy = wrf.Train(dataset, labels, 3, weights, 10, 1);

  REQUIRE(std::isfinite(entropy) == true);
}

/**
 * Test that RandomForest::Train() returns finite average entropy on categorical
 * dataset.
 */
TEST_CASE("RandomForestCategoricalTrainReturnEntropy", "[RandomForestTest]")
{
  arma::mat d;
  arma::Row<size_t> l;
  data::DatasetInfo di;
  MockCategoricalData(d, l, di);

  // Now create random points.
  arma::mat randomNoise(4, 2000);
  arma::Row<size_t> randomLabels(2000);
  for (size_t i = 0; i < 2000; ++i)
  {
    randomNoise(0, i) = Random();
    randomNoise(1, i) = Random();
    randomNoise(2, i) = RandInt(4);
    randomNoise(3, i) = RandInt(2);
    randomLabels[i] = RandInt(5);
  }

  // Generate weights.
  arma::rowvec weights(6000);
  for (size_t i = 0; i < 4000; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = 4000; i < 6000; ++i)
    weights[i] = Random(0.0, 0.001);

  arma::mat fullData = arma::join_rows(d, randomNoise);
  arma::Row<size_t> fullLabels = arma::join_rows(l, randomLabels);

  // Test random forest on unweighted categorical dataset.
  RandomForest<> rf;
  double entropy = rf.Train(fullData, di, fullLabels, 5, 15 /* 15 trees */, 1,
      1e-7, 0, false, MultipleRandomDimensionSelect(3));

  REQUIRE(std::isfinite(entropy) == true);

  // Test random forest on weighted categorical dataset.
  RandomForest<> wrf;
  entropy = wrf.Train(fullData, di, fullLabels, 5, weights, 15 /* 15 trees */,
      1, 1e-7, 0, false, MultipleRandomDimensionSelect(3));

  REQUIRE(std::isfinite(entropy) == true);
}

/**
 * Test that different trees get generated.
 */
TEST_CASE("DifferentTreesTest", "[RandomForestTest]")
{
  arma::mat d(10, 100, arma::fill::randu);
  arma::Row<size_t> l(100);
  for (size_t i = 0; i < 50; ++i)
    l(i) = 0;
  for (size_t i = 50; i < 100; ++i)
    l(i) = 1;

  bool success = false;
  size_t trial = 0;

  // It's possible we might get the same random dimensions selected, so let's do
  // multiple trials.
  while (!success && trial < 5)
  {
    RandomForest<GiniGain, RandomDimensionSelect> rf;
    rf.Train(d, l, 2, 2, 5);

    success = (rf.Tree(0).SplitDimension() != rf.Tree(1).SplitDimension());

    ++trial;
  }

  REQUIRE(success == true);
}

/**
 * Test that RandomForest::Train() when passed warmStart = True trains on top
 * of exixting forest and adds the newly trained trees to the previously
 * exixting forest.
 */
TEST_CASE("WarmStartTreesTest", "[RandomForestTest]")
{
  arma::mat trainingData;
  arma::Row<size_t> trainingLabels;
  data::DatasetInfo di;
  MockCategoricalData(trainingData, trainingLabels, di);

  // Train a random forest.
  RandomForest<> rf(trainingData, di, trainingLabels, 5, 25 /* 25 trees */, 1,
      1e-7, 0, MultipleRandomDimensionSelect(4));

  REQUIRE(rf.NumTrees() == 25);

  rf.Train(trainingData, di, trainingLabels, 5, 20 /* 20 trees */, 1, 1e-7, 0,
      true /* warmStart */, MultipleRandomDimensionSelect(4));

  REQUIRE(rf.NumTrees() == 25 + 20);
}

/**
 * Test that RandomForest::Train() when passed warmStart = True does not drop
 * prediction quality on train data. Note that prediction quality may drop due
 * to overfitting in some cases.
 */
TEST_CASE("WarmStartTreesPredictionsQualityTest", "[RandomForestTest]")
{
  arma::mat trainingData;
  arma::Row<size_t> trainingLabels;
  data::DatasetInfo di;
  MockCategoricalData(trainingData, trainingLabels, di);

  // Train a random forest.
  RandomForest<> rf(trainingData, di, trainingLabels, 5, 3 /* 3 trees */, 1,
      1e-7, 0, MultipleRandomDimensionSelect(4));

  // Get performance statistics on train data.
  arma::Row<size_t> oldPredictions;
  rf.Classify(trainingData, oldPredictions);

  // Calculate the number of correct points.
  size_t oldCorrect = arma::accu(oldPredictions == trainingLabels);

  rf.Train(trainingData, di, trainingLabels, 5, 20 /* 20 trees */, 1, 1e-7, 0,
      true /* warmStart */, MultipleRandomDimensionSelect(4));

  // Get performance statistics on train data.
  arma::Row<size_t> newPredictions;
  rf.Classify(trainingData, newPredictions);

  // Calculate the number of correct points.
  size_t newCorrect = arma::accu(newPredictions == trainingLabels);

  REQUIRE(newCorrect >= oldCorrect);
}

/**
 * Ensure that the Extra Trees algorithm gives decent accuracy.
 */
TEST_CASE("ExtraTreesAccuracyTest", "[RandomForestTest]")
{
  // Load the iris dataset.
  arma::mat dataset;
  if (!data::Load("iris_train.csv", dataset))
    FAIL("Cannot load dataset iris_train.csv");
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load dataset iris_train_labels.csv");

  // Add some noise.
  arma::mat noise(dataset.n_rows, 1000, arma::fill::randu);
  arma::Row<size_t> noiseLabels(1000);
  for (size_t i = 0; i < noiseLabels.n_elem; ++i)
    noiseLabels[i] = RandInt(3); // Random label.

  // Concatenate data matrices.
  arma::mat data = arma::join_rows(dataset, noise);
  arma::Row<size_t> fullLabels = arma::join_rows(labels, noiseLabels);

  // Now set weights.
  arma::rowvec weights(dataset.n_cols + 1000);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = dataset.n_cols; i < dataset.n_cols + 1000; ++i)
    weights[i] = Random(0.0, 0.01); // Low weights for false points.

  // Train extra tree.
  ExtraTrees<> et(data, fullLabels, 3, weights, 20, 1);

  // Get performance statistics on test data.
  arma::mat testDataset;
  if (!data::Load("iris_test.csv", testDataset))
    FAIL("Cannot load dataset iris_test.csv");
  arma::Row<size_t> testLabels;
  if (!data::Load("iris_test_labels.csv", testLabels))
    FAIL("Cannot load dataset iris_test_labels.csv");

  arma::Row<size_t> predictions;
  et.Classify(testDataset, predictions);

  // Calculate the prediction accuracy.
  double accuracy = arma::accu(predictions == testLabels);
  accuracy /= predictions.n_elem;

  REQUIRE(accuracy >= 0.91);
}
