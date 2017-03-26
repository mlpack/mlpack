/**
 * @file decision_tree_test.cpp
 * @author Ryan Curtin
 *
 * Tests for the DecisionTree class and related classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(DecisionTreeTest);

/**
 * Make sure the Gini gain is zero when the labels are perfect.
 */
BOOST_AUTO_TEST_CASE(GiniGainPerfectTest)
{
  arma::Row<size_t> labels;
  labels.zeros(10);

  // Test that it's perfect regardless of number of classes.
  for (size_t c = 1; c < 10; ++c)
    BOOST_REQUIRE_SMALL(GiniGain::Evaluate(labels, c), 1e-5);
}

/**
 * Make sure the Gini gain is -0.5 when the class split between two classes
 * is even.
 */
BOOST_AUTO_TEST_CASE(GiniGainEvenSplitTest)
{
  arma::Row<size_t> labels(10);
  for (size_t i = 0; i < 5; ++i)
    labels[i] = 0;
  for (size_t i = 5; i < 10; ++i)
    labels[i] = 1;

  // Test that it's -0.5 regardless of the number of classes.
  for (size_t c = 2; c < 10; ++c)
    BOOST_REQUIRE_CLOSE(GiniGain::Evaluate(labels, c), -0.5, 1e-5);
}

/**
 * The Gini gain of an empty vector is 0.
 */
BOOST_AUTO_TEST_CASE(GiniGainEmptyTest)
{
  // Test across some numbers of classes.
  arma::Row<size_t> labels;
  for (size_t c = 1; c < 10; ++c)
    BOOST_REQUIRE_SMALL(GiniGain::Evaluate(labels, c), 1e-5);
}

/**
 * The Gini gain is -(1 - 1/k) for k classes evenly split.
 */
BOOST_AUTO_TEST_CASE(GiniGainEvenSplitManyClassTest)
{
  // Try with many different classes.
  for (size_t c = 2; c < 30; ++c)
  {
    arma::Row<size_t> labels(c);
    for (size_t i = 0; i < c; ++i)
      labels[i] = i;

    // Calculate Gini gain and make sure it is correct.
    BOOST_REQUIRE_CLOSE(GiniGain::Evaluate(labels, c), -(1.0 - 1.0 / c), 1e-5);
  }
}

/**
 * The Gini gain should not be sensitive to the number of points.
 */
BOOST_AUTO_TEST_CASE(GiniGainManyPoints)
{
  for (size_t i = 1; i < 20; ++i)
  {
    const size_t numPoints = 100 * i;
    arma::Row<size_t> labels(numPoints);
    for (size_t j = 0; j < numPoints / 2; ++j)
      labels[j] = 0;
    for (size_t j = numPoints / 2; j < numPoints; ++j)
      labels[j] = 1;

    BOOST_REQUIRE_CLOSE(GiniGain::Evaluate(labels, 2), -0.5, 1e-5);
  }
}

/**
 * The information gain should be zero when the labels are perfect.
 */
BOOST_AUTO_TEST_CASE(InformationGainPerfectTest)
{
  arma::Row<size_t> labels;
  labels.zeros(10);

  // Test that it's perfect regardless of number of classes.
  for (size_t c = 1; c < 10; ++c)
    BOOST_REQUIRE_SMALL(InformationGain::Evaluate(labels, c), 1e-5);
}

/**
 * If we have an even split, the information gain should be -1.
 */
BOOST_AUTO_TEST_CASE(InformationGainEvenSplitTest)
{
  arma::Row<size_t> labels(10);
  for (size_t i = 0; i < 5; ++i)
    labels[i] = 0;
  for (size_t i = 5; i < 10; ++i)
    labels[i] = 1;

  // Test that it's -1 regardless of the number of classes.
  for (size_t c = 2; c < 10; ++c)
    BOOST_REQUIRE_CLOSE(InformationGain::Evaluate(labels, c), -1.0, 1e-5);
}

/**
 * The information gain of an empty vector is 0.
 */
BOOST_AUTO_TEST_CASE(InformationGainEmptyTest)
{
  arma::Row<size_t> labels;
  for (size_t c = 1; c < 10; ++c)
    BOOST_REQUIRE_SMALL(InformationGain::Evaluate(labels, c), 1e-5);
}

/**
 * The information gain is log2(1/k) when splitting equal classes.
 */
BOOST_AUTO_TEST_CASE(InformationGainEvenSplitManyClassTest)
{
  // Try with many different numbers of classes.
  for (size_t c = 2; c < 30; ++c)
  {
    arma::Row<size_t> labels(c);
    for (size_t i = 0; i < c; ++i)
      labels[i] = i;

    // Calculate information gain and make sure it is correct.
    BOOST_REQUIRE_CLOSE(InformationGain::Evaluate(labels, c),
        std::log2(1.0 / c), 1e-5);
  }
}

/**
 * The information gain should not be sensitive to the number of points.
 */
BOOST_AUTO_TEST_CASE(InformationGainManyPoints)
{
  for (size_t i = 1; i < 20; ++i)
  {
    const size_t numPoints = 100 * i;
    arma::Row<size_t> labels(numPoints);
    for (size_t j = 0; j < numPoints / 2; ++j)
      labels[j] = 0;
    for (size_t j = numPoints / 2; j < numPoints; ++j)
      labels[j] = 1;

    BOOST_REQUIRE_CLOSE(InformationGain::Evaluate(labels, 2), -1.0, 1e-5);
  }
}

/**
 * Check that the BestBinaryNumericSplit will split on an obviously splittable
 * dimension.
 */
BOOST_AUTO_TEST_CASE(BestBinaryNumericSplitSimpleSplitTest)
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::Row<size_t> labels("0 0 0 0 0 1 1 1 1 1 1");

  arma::vec classProbabilities;
  BestBinaryNumericSplit<GiniGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate(labels, 2);
  const double gain = BestBinaryNumericSplit<GiniGain>::SplitIfBetter(bestGain,
      values, labels, 2, 3, classProbabilities, aux);

  // Make sure that a split was made.
  BOOST_REQUIRE_GT(gain, bestGain);

  // The split is perfect, so we should be able to accomplish a gain of 0.
  BOOST_REQUIRE_SMALL(gain, 1e-5);

  // The class probabilities, for this split, hold the splitting point, which
  // should be between 4 and 5.
  BOOST_REQUIRE_EQUAL(classProbabilities.n_elem, 1);
  BOOST_REQUIRE_GT(classProbabilities[0], 0.4);
  BOOST_REQUIRE_LT(classProbabilities[0], 0.5);
}

/**
 * Check that the BestBinaryNumericSplit won't split if not enough points are
 * given.
 */
BOOST_AUTO_TEST_CASE(BestBinaryNumericSplitMinSamplesTest)
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::Row<size_t> labels("0 0 0 0 0 1 1 1 1 1 1");

  arma::vec classProbabilities;
  BestBinaryNumericSplit<GiniGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate(labels, 2);
  const double gain = BestBinaryNumericSplit<GiniGain>::SplitIfBetter(bestGain,
      values, labels, 2, 8, classProbabilities, aux);

  // Make sure that no split was made.
  BOOST_REQUIRE_EQUAL(gain, bestGain);
  BOOST_REQUIRE_EQUAL(classProbabilities.n_elem, 0);
}

/**
 * Check that the BestBinaryNumericSplit doesn't split a dimension that gives no
 * gain.
 */
BOOST_AUTO_TEST_CASE(BestBinaryNumericSplitNoGainTest)
{
  arma::vec values(100);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < 100; i += 2)
  {
    values[i] = i;
    labels[i] = 0;
    values[i + 1] = i;
    labels[i + 1] = 1;
  }

  arma::vec classProbabilities;
  BestBinaryNumericSplit<GiniGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate(labels, 2);
  const double gain = BestBinaryNumericSplit<GiniGain>::SplitIfBetter(bestGain,
      values, labels, 2, 10, classProbabilities, aux);

  // Make sure there was no split.
  BOOST_REQUIRE_EQUAL(gain, bestGain);
  BOOST_REQUIRE_EQUAL(classProbabilities.n_elem, 0);
}

/**
 * Check that the AllCategoricalSplit will split when the split is obviously
 * better.
 */
BOOST_AUTO_TEST_CASE(AllCategoricalSplitSimpleSplitTest)
{
  arma::vec values("0 0 0 1 1 1 2 2 2 3 3 3");
  arma::Row<size_t> labels("0 0 0 2 2 2 1 1 1 2 2 2");

  arma::vec classProbabilities;
  AllCategoricalSplit<GiniGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate(labels, 3);
  const double gain = AllCategoricalSplit<GiniGain>::SplitIfBetter(bestGain,
      values, 4, labels, 3, 3, classProbabilities, aux);

  // Make sure that a split was made.
  BOOST_REQUIRE_GT(gain, bestGain);

  // Since the split is perfect, make sure the new gain is 0.
  BOOST_REQUIRE_SMALL(gain, 1e-5);

  // Make sure the class probabilities now hold the number of children.
  BOOST_REQUIRE_EQUAL(classProbabilities.n_elem, 1);
  BOOST_REQUIRE_EQUAL((size_t) classProbabilities[0], 4);
}

/**
 * Make sure that AllCategoricalSplit respects the minimum number of samples
 * required to split.
 */
BOOST_AUTO_TEST_CASE(AllCategoricalSplitMinSamplesTest)
{
  arma::vec values("0 0 0 1 1 1 2 2 2 3 3 3");
  arma::Row<size_t> labels("0 0 0 2 2 2 1 1 1 2 2 2");

  arma::vec classProbabilities;
  AllCategoricalSplit<GiniGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate(labels, 3);
  const double gain = AllCategoricalSplit<GiniGain>::SplitIfBetter(bestGain,
      values, 4, labels, 3, 4, classProbabilities, aux);

  // Make sure it's not split.
  BOOST_REQUIRE_EQUAL(gain, bestGain);
  BOOST_REQUIRE_EQUAL(classProbabilities.n_elem, 0);
}

/**
 * Check that no split is made when it doesn't get us anything.
 */
BOOST_AUTO_TEST_CASE(AllCategoricalSplitNoGainTest)
{
  arma::vec values(300);
  arma::Row<size_t> labels(300);
  for (size_t i = 0; i < 300; i += 3)
  {
    values[i] = (i / 3) % 10;
    labels[i] = 0;
    values[i + 1] = (i / 3) % 10;
    labels[i + 1] = 1;
    values[i + 2] = (i / 3) % 10;
    labels[i + 2] = 2;
  }

  arma::vec classProbabilities;
  AllCategoricalSplit<GiniGain>::template AuxiliarySplitInfo<double> aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate(labels, 3);
  const double gain = AllCategoricalSplit<GiniGain>::SplitIfBetter(bestGain,
      values, 10, labels, 3, 10, classProbabilities, aux);

  // Make sure that there was no split.
  BOOST_REQUIRE_EQUAL(gain, bestGain);
  BOOST_REQUIRE_EQUAL(classProbabilities.n_elem, 0);
}

/**
 * A basic construction of the decision tree---ensure that we can create the
 * tree and that it split at least once.
 */
BOOST_AUTO_TEST_CASE(BasicConstructionTest)
{
  arma::mat dataset(10, 1000, arma::fill::randu);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 1000; ++i)
    labels[i] = i % 3; // 3 classes.

  // Use default parameters.
  DecisionTree<> d(dataset, labels, 3, 50);

  // Now require that we have some children.
  BOOST_REQUIRE_GT(d.NumChildren(), 0);
}

/**
 * Construct the decision tree on numeric data only and see that we can fit it
 * exactly and achieve perfect performance on the training set.
 */
BOOST_AUTO_TEST_CASE(PerfectTrainingSet)
{
  // Completely random dataset with no structure.
  arma::mat dataset(10, 1000, arma::fill::randu);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 1000; ++i)
    labels[i] = i % 3; // 3 classes.

  DecisionTree<> d(dataset, labels, 3, 1); // Minimum leaf size of 1.

  // Make sure that we can get perfect accuracy on the training set.
  for (size_t i = 0; i < 1000; ++i)
  {
    size_t prediction;
    arma::vec probabilities;
    d.Classify(dataset.col(i), prediction, probabilities);

    BOOST_REQUIRE_EQUAL(prediction, labels[i]);
    BOOST_REQUIRE_EQUAL(probabilities.n_elem, 3);
    for (size_t j = 0; j < 3; ++j)
    {
      if (labels[i] == j)
        BOOST_REQUIRE_CLOSE(probabilities[j], 1.0, 1e-5);
      else
        BOOST_REQUIRE_SMALL(probabilities[j], 1e-5);
    }
  }
}

/**
 * Make sure class probabilities are computed correctly in the root node.
 */
BOOST_AUTO_TEST_CASE(ClassProbabilityTest)
{
  arma::mat dataset(5, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < 100; i += 2)
  {
    labels[i] = 0;
    labels[i + 1] = 1;
  }

  // Create a decision tree that can't split.
  DecisionTree<> d(dataset, labels, 2, 1000);

  BOOST_REQUIRE_EQUAL(d.NumChildren(), 0);

  // Estimate a point's probabilities.
  arma::vec probabilities;
  size_t prediction;
  d.Classify(dataset.col(0), prediction, probabilities);

  BOOST_REQUIRE_EQUAL(probabilities.n_elem, 2);
  BOOST_REQUIRE_CLOSE(probabilities[0], 0.5, 1e-5);
  BOOST_REQUIRE_CLOSE(probabilities[1], 0.5, 1e-5);
}

/**
 * Test that the decision tree generalizes reasonably.
 */
BOOST_AUTO_TEST_CASE(SimpleGeneralizationTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Build decision tree.
  DecisionTree<> d(inputData, labels, 3, 10); // Leaf size of 10.

  // Load testing data.
  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset vc2_test.csv!");

  arma::Mat<size_t> trueTestLabels;
  if (!data::Load("vc2_test_labels.txt", trueTestLabels))
    BOOST_FAIL("Cannot load labels for vc2_test_labels.txt");

  // Get the predicted test labels.
  arma::Row<size_t> predictions;
  d.Classify(testData, predictions);

  BOOST_REQUIRE_EQUAL(predictions.n_elem, testData.n_cols);

  // Figure out the accuracy.
  double correct = 0.0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions[i] == trueTestLabels[i])
      ++correct;
  correct /= predictions.n_elem;

  BOOST_REQUIRE_GT(correct, 0.75);
}

/**
 * Test that we can build a decision tree on a simple categorical dataset.
 */
BOOST_AUTO_TEST_CASE(CategoricalBuildTest)
{
  // We'll build a spiral dataset plus two noisy categorical features.  We need
  // to build the distributions for the categorical features (they'll be
  // discrete distributions).
  DiscreteDistribution c1[5];
  // The distribution will be automatically normalized.
  for (size_t i = 0; i < 5; ++i)
  {
    std::vector<arma::vec> probs;
    probs.push_back(arma::vec(4, arma::fill::randu));
    c1[i] = DiscreteDistribution(probs);
  }

  DiscreteDistribution c2[5];
  for (size_t i = 0; i < 5; ++i)
  {
    std::vector<arma::vec> probs;
    probs.push_back(arma::vec(2, arma::fill::randu));
    c2[i] = DiscreteDistribution(probs);
  }

  arma::mat spiralDataset(4, 10000);
  arma::Row<size_t> labels(10000);
  for (size_t i = 0; i < 10000; ++i)
  {
    // One circle every 20000 samples.  Plus some noise.
    const double magnitude = 2.0 + (double(i) / 2000.0) +
        0.5 * mlpack::math::Random();
    const double angle = (i % 2000) * (2 * M_PI) + mlpack::math::Random();

    const double x = magnitude * cos(angle);
    const double y = magnitude * sin(angle);

    spiralDataset(0, i) = x;
    spiralDataset(1, i) = y;

    // Set categorical features c1 and c2.
    if (i < 2000)
    {
      spiralDataset(2, i) = c1[1].Random()[0];
      spiralDataset(3, i) = c2[1].Random()[0];
      labels[i] = 1;
    }
    else if (i < 4000)
    {
      spiralDataset(2, i) = c1[3].Random()[0];
      spiralDataset(3, i) = c2[3].Random()[0];
      labels[i] = 3;
    }
    else if (i < 6000)
    {
      spiralDataset(2, i) = c1[2].Random()[0];
      spiralDataset(3, i) = c2[2].Random()[0];
      labels[i] = 2;
    }
    else if (i < 8000)
    {
      spiralDataset(2, i) = c1[0].Random()[0];
      spiralDataset(3, i) = c2[0].Random()[0];
      labels[i] = 0;
    }
    else
    {
      spiralDataset(2, i) = c1[4].Random()[0];
      spiralDataset(3, i) = c2[4].Random()[0];
      labels[i] = 4;
    }
  }

  // Now create the dataset info.
  data::DatasetInfo di(4);
  di.Type(2) = data::Datatype::categorical;
  di.Type(3) = data::Datatype::categorical;
  // Set mappings.
  di.MapString<double>("0", 2);
  di.MapString<double>("1", 2);
  di.MapString<double>("2", 2);
  di.MapString<double>("3", 2);
  di.MapString<double>("0", 3);
  di.MapString<double>("1", 3);

  // Now shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, 9999,
      10000));
  arma::mat d(4, 10000);
  arma::Row<size_t> l(10000);
  for (size_t i = 0; i < 10000; ++i)
  {
    d.col(i) = spiralDataset.col(indices[i]);
    l[i] = labels[indices[i]];
  }

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 4999);
  arma::mat testData = d.cols(5000, 9999);
  arma::Row<size_t> trainingLabels = l.subvec(0, 4999);
  arma::Row<size_t> testLabels = l.subvec(5000, 9999);

  // Build the tree.
  DecisionTree<> tree(trainingData, di, trainingLabels, 5, 10);

  // Now evaluate the accuracy of the tree.
  arma::Row<size_t> predictions;
  tree.Classify(testData, predictions);

  BOOST_REQUIRE_EQUAL(predictions.n_elem, testData.n_cols);
  size_t correct = 0;
  for (size_t i = 0; i < testData.n_cols; ++i)
    if (testLabels[i] == predictions[i])
      ++correct;

  // Make sure we got at least 70% accuracy.
  const double correctPct = double(correct) / double(testData.n_cols);
  BOOST_REQUIRE_GT(correctPct, 0.70);
}

/**
 * Make sure that when we ask for a decision stump, we get one.
 */
BOOST_AUTO_TEST_CASE(DecisionStumpTest)
{
  // Use a random dataset.
  arma::mat dataset(10, 1000, arma::fill::randu);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 1000; ++i)
    labels[i] = i % 3; // 3 classes.

  // Build a decision stump.
  DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, double,
      true> stump(dataset, labels, 3, 1);

  // Check that it has children.
  BOOST_REQUIRE_EQUAL(stump.NumChildren(), 2);
  // Check that its children doesn't have children.
  BOOST_REQUIRE_EQUAL(stump.Child(0).NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(stump.Child(1).NumChildren(), 0);
}

BOOST_AUTO_TEST_SUITE_END();
