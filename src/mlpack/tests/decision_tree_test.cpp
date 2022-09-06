/**
 * @file tests/decision_tree_test.cpp
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
#include <mlpack/methods/decision_tree.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"

using namespace mlpack;

/**
 * Make sure the Gini gain is zero when the labels are perfect.
 */
TEST_CASE("GiniGainPerfectTest", "[DecisionTreeTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::Row<size_t> labels;
  labels.zeros(10);

  // Test that it's perfect regardless of number of classes.
  for (size_t c = 1; c < 10; ++c)
  {
    REQUIRE(GiniGain::Evaluate<false>(labels, c, weights) ==
        Approx(0.0).margin(1e-5));
  }
}

/**
 * Make sure the Gini gain is -0.5 when the class split between two classes
 * is even.
 */
TEST_CASE("GiniGainEvenSplitTest", "[DecisionTreeTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::Row<size_t> labels(10);
  for (size_t i = 0; i < 5; ++i)
    labels[i] = 0;
  for (size_t i = 5; i < 10; ++i)
    labels[i] = 1;

  // Test that it's -0.5 regardless of the number of classes.
  for (size_t c = 2; c < 10; ++c)
  {
    REQUIRE(GiniGain::Evaluate<false>(labels, c, weights) ==
        Approx(-0.5).epsilon(1e-7));

    double weightedGain = GiniGain::Evaluate<true>(labels, c, weights);

    // The weighted gain should stay the same with unweight one
    REQUIRE(GiniGain::Evaluate<false>(labels, c, weights) == weightedGain);
  }
}

/**
 * The Gini gain of an empty vector is 0.
 */
TEST_CASE("GiniGainEmptyTest", "[DecisionTreeTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  // Test across some numbers of classes.
  arma::Row<size_t> labels;
  for (size_t c = 1; c < 10; ++c)
  {
    REQUIRE(GiniGain::Evaluate<false>(labels, c, weights) ==
        Approx(0.0).margin(1e-5));
  }

  for (size_t c = 1; c < 10; ++c)
  {
    REQUIRE(GiniGain::Evaluate<true>(labels, c, weights) ==
        Approx(0.0).margin(1e-5));
  }
}

/**
 * The Gini gain is -(1 - 1/k) for k classes evenly split.
 */
TEST_CASE("GiniGainEvenSplitManyClassTest", "[DecisionTreeTest]")
{
  // Try with many different classes.
  for (size_t c = 2; c < 30; ++c)
  {
    arma::Row<size_t> labels(c);
    arma::rowvec weights(c);
    for (size_t i = 0; i < c; ++i)
    {
      labels[i] = i;
      weights[i] = 1;
    }

    // Calculate Gini gain and make sure it is correct.
    REQUIRE(GiniGain::Evaluate<false>(labels, c, weights) ==
        Approx(-(1.0 - 1.0 / c)).epsilon(1e-7));
    REQUIRE(GiniGain::Evaluate<true>(labels, c, weights) ==
        Approx(-(1.0 - 1.0 / c)).epsilon(1e-7));
  }
}

/**
 * The Gini gain should not be sensitive to the number of points.
 */
TEST_CASE("GiniGainManyPoints", "[DecisionTreeTest]")
{
  for (size_t i = 1; i < 20; ++i)
  {
    const size_t numPoints = 100 * i;
    arma::rowvec weights(numPoints);
    weights.ones();
    arma::Row<size_t> labels(numPoints);
    for (size_t j = 0; j < numPoints / 2; ++j)
      labels[j] = 0;
    for (size_t j = numPoints / 2; j < numPoints; ++j)
      labels[j] = 1;
    REQUIRE(GiniGain::Evaluate<false>(labels, 2, weights) ==
        Approx(-0.5).epsilon(1e-7));
    REQUIRE(GiniGain::Evaluate<true>(labels, 2, weights) ==
        Approx(-0.5).epsilon(1e-7));
  }
}


/**
 * To make sure the Gini gain can been cacluate proporately with weight.
 */
TEST_CASE("GiniGainWithWeight", "[DecisionTreeTest]")
{
  arma::Row<size_t> labels(10);
  arma::rowvec weights(10);
  for (size_t i = 0; i < 5; ++i)
  {
    labels[i] = 0;
    weights[i] = 0.3;
  }
  for (size_t i = 5; i < 10; ++i)
  {
    labels[i] = 1;
    weights[i] = 0.7;
  }

  REQUIRE(GiniGain::Evaluate<true>(labels, 2, weights) ==
      Approx(-0.42).epsilon(1e-7));
}

/**
 * The information gain should be zero when the labels are perfect.
 */
TEST_CASE("InformationGainPerfectTest", "[DecisionTreeTest]")
{
  arma::rowvec weights;
  arma::Row<size_t> labels;
  labels.zeros(10);

  // Test that it's perfect regardless of number of classes.
  for (size_t c = 1; c < 10; ++c)
  {
    REQUIRE(InformationGain::Evaluate<false>(labels, c, weights) ==
        Approx(0.0).margin(1e-5));
  }
}

/**
 * If we have an even split, the information gain should be -1.
 */
TEST_CASE("InformationGainEvenSplitTest", "[DecisionTreeTest]")
{
  arma::Row<size_t> labels(10);
  arma::rowvec weights(10);
  weights.ones();
  for (size_t i = 0; i < 5; ++i)
    labels[i] = 0;
  for (size_t i = 5; i < 10; ++i)
    labels[i] = 1;

  // Test that it's -1 regardless of the number of classes.
  for (size_t c = 2; c < 10; ++c)
  {
    // Weighted and unweighted result should be the same.
    REQUIRE(InformationGain::Evaluate<false>(labels, c, weights) ==
        Approx(-1.0).epsilon(1e-7));
    REQUIRE(InformationGain::Evaluate<true>(labels, c, weights) ==
        Approx(-1.0).epsilon(1e-7));
  }
}

/**
 * The information gain of an empty vector is 0.
 */
TEST_CASE("InformationGainEmptyTest", "[DecisionTreeTest]")
{
  arma::Row<size_t> labels;
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  for (size_t c = 1; c < 10; ++c)
  {
    REQUIRE(InformationGain::Evaluate<false>(labels, c, weights) ==
        Approx(0.0).margin(1e-5));
    REQUIRE(InformationGain::Evaluate<true>(labels, c, weights) ==
        Approx(0.0).margin(1e-5));
  }
}

/**
 * The information gain is log2(1/k) when splitting equal classes.
 */
TEST_CASE("InformationGainEvenSplitManyClassTest", "[DecisionTreeTest]")
{
  arma::rowvec weights;
  // Try with many different numbers of classes.
  for (size_t c = 2; c < 30; ++c)
  {
    arma::Row<size_t> labels(c);
    for (size_t i = 0; i < c; ++i)
      labels[i] = i;

    // Calculate information gain and make sure it is correct.
    REQUIRE(InformationGain::Evaluate<false>(labels, c, weights) ==
        Approx(std::log2(1.0 / c)).epsilon(1e-7));
  }
}

/**
 * Test the information gain with weighted labels
 */
TEST_CASE("InformationWithWeight", "[DecisionTreeTest]")
{
  arma::Row<size_t> labels(10);
  arma::rowvec weights("1 1 1 1 1 0 0 0 0 0");
  for (size_t i = 0; i < 5; ++i)
    labels[i] = 0;
  for (size_t i = 5; i < 10; ++i)
    labels[i] = 1;

  // Zero is not a good result as gain, but we just need to prove
  // cacluation works.
  REQUIRE(InformationGain::Evaluate<true>(labels, 2, weights) ==
      Approx(0).epsilon(1e-7));
}


/**
 * The information gain should not be sensitive to the number of points.
 */
TEST_CASE("InformationGainManyPoints", "[DecisionTreeTest]")
{
  for (size_t i = 1; i < 20; ++i)
  {
    const size_t numPoints = 100 * i;
    arma::Row<size_t> labels(numPoints);
    arma::rowvec weights = arma::ones<arma::rowvec>(numPoints);
    for (size_t j = 0; j < numPoints / 2; ++j)
      labels[j] = 0;
    for (size_t j = numPoints / 2; j < numPoints; ++j)
      labels[j] = 1;

    REQUIRE(InformationGain::Evaluate<false>(labels, 2, weights) ==
        Approx(-1.0).epsilon(1e-7));

    // It should make no difference between a weighted and unweighted
    // calculation.
    REQUIRE(InformationGain::Evaluate<true>(labels, 2, weights) ==
        Approx(-1.0).epsilon(1e-7));
  }
}

/**
 * Check that the BestBinaryNumericSplit will split on an obviously splittable
 * dimension.
 */
TEST_CASE("BestBinaryNumericSplitSimpleSplitTest", "[DecisionTreeTest]")
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::Row<size_t> labels("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  arma::vec classProbabilities;
  BestBinaryNumericSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 2, weights);
  const double gain = BestBinaryNumericSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, labels, 2, weights, 3, 1e-7, classProbabilities, aux);
  const double weightedGain =
      BestBinaryNumericSplit<GiniGain>::SplitIfBetter<true>(bestGain, values,
      labels, 2, weights, 3, 1e-7, classProbabilities, aux);

  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  // Make sure weight works and is not different than the unweighted one.
  REQUIRE(gain == weightedGain);

  // The split is perfect, so we should be able to accomplish a gain of 0.
  REQUIRE(gain == Approx(0.0).margin(1e-7));

  // The class probabilities, for this split, hold the splitting point, which
  // should be between 4 and 5.
  REQUIRE(classProbabilities.n_elem == 1);
  REQUIRE(classProbabilities[0] > 0.4);
  REQUIRE(classProbabilities[0] < 0.5);
}

/**
 * Check that the BestBinaryNumericSplit won't split if not enough points are
 * given.
 */
TEST_CASE("BestBinaryNumericSplitMinSamplesTest", "[DecisionTreeTest]")
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::Row<size_t> labels("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights(labels.n_elem);

  arma::vec classProbabilities;
  BestBinaryNumericSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 2, weights);
  const double gain = BestBinaryNumericSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, labels, 2, weights, 8, 1e-7, classProbabilities,
      aux);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      BestBinaryNumericSplit<GiniGain>::SplitIfBetter<true>(bestGain, values,
      labels, 2, weights, 8, 1e-7, classProbabilities, aux);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
  REQUIRE(classProbabilities.n_elem == 0);
}

/**
 * Check that the BestBinaryNumericSplit doesn't split a dimension that gives no
 * gain.
 */
TEST_CASE("BestBinaryNumericSplitNoGainTest", "[DecisionTreeTest]")
{
  arma::vec values(100);
  arma::Row<size_t> labels(100);
  arma::rowvec weights;
  for (size_t i = 0; i < 100; i += 2)
  {
    values[i] = i;
    labels[i] = 0;
    values[i + 1] = i;
    labels[i + 1] = 1;
  }

  arma::vec classProbabilities;
  BestBinaryNumericSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 2, weights);
  const double gain = BestBinaryNumericSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, labels, 2, weights, 10, 1e-7, classProbabilities,
      aux);

  // Make sure there was no split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(classProbabilities.n_elem == 0);
}

/**
 * Check that the RandomBinaryNumericSplit won't split if not enough points are
 * given.
 */
TEST_CASE("RandomBinaryNumericSplitMinSamplesTest", "[DecisionTreeTest]")
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::Row<size_t> labels("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights(labels.n_elem);

  arma::vec classProbabilities(1);
  RandomBinaryNumericSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 2, weights);
  const double gain = RandomBinaryNumericSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, labels, 2, weights, 8, 1e-7, classProbabilities, aux);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      RandomBinaryNumericSplit<GiniGain>::SplitIfBetter<true>(bestGain, values,
      labels, 2, weights, 8, 1e-7, classProbabilities, aux);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
}

/**
 * Check that the RandomBinaryNumericSplit doesn't split a dimension that gives
 * no gain when splitIfBetterGain is true.
 */
TEST_CASE("RandomBinaryNumericSplitNoGainTest", "[DecisionTreeTest]")
{
  arma::vec values(100);
  arma::Row<size_t> labels(100);
  arma::rowvec weights;
  for (size_t i = 0; i < 100; i += 2)
  {
    values[i] = i;
    labels[i] = 0;
    values[i + 1] = i;
    labels[i + 1] = 1;
  }

  arma::vec classProbabilities;
  RandomBinaryNumericSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 2, weights);
  const double gain = RandomBinaryNumericSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, labels, 2, weights, 10, 1e-7, classProbabilities,
      aux, true);

  // Make sure there was no split.
  REQUIRE(gain == DBL_MAX);
}

/**
 * Check that RandomBinaryNumericSplit generally gives a split different than
 * the BestBinaryNumericSplit.
 */
TEST_CASE("RandomBinaryNumericSplitDiffSplitTest", "[DecisionTreeTest]")
{
  arma::vec values(1000);
  arma::Row<size_t> labels(1000);
  arma::rowvec weights;
  for (size_t i = 0; i < 1000; i += 2)
  {
    values[i] = Random(0, 5);
    labels[i] = 0;
    values[i + 1] = Random(0, 5);
    labels[i + 1] = 1;
  }

  arma::vec classProbabilities, classProbabilities1;
  BestBinaryNumericSplit<GiniGain>::AuxiliarySplitInfo aux;
  RandomBinaryNumericSplit<GiniGain>::AuxiliarySplitInfo aux1;

  const double bestGain = GiniGain::Evaluate<false>(labels, 2, weights);

  for (int i = 0; i < 5; ++i)
  {
    // Call BestBinaryNumericSplit to do the splitting.
    (void) BestBinaryNumericSplit<GiniGain>::SplitIfBetter<false>(
        bestGain, values, labels, 2, weights, 3, 1e-7, classProbabilities,
        aux);

    // Call RandomBinaryNumericSplit to do the splitting.
    (void) RandomBinaryNumericSplit<GiniGain>::SplitIfBetter<false>(
        bestGain, values, labels, 2, weights, 3, 1e-7, classProbabilities1,
        aux1);

    if (classProbabilities[0] == classProbabilities1[0])
      break;
  }

  REQUIRE(classProbabilities[0] != classProbabilities1[0]);
}

/**
 * Check that the AllCategoricalSplit will split when the split is obviously
 * better.
 */
TEST_CASE("AllCategoricalSplitSimpleSplitTest", "[DecisionTreeTest]")
{
  arma::vec values("0 0 0 1 1 1 2 2 2 3 3 3");
  arma::Row<size_t> labels("0 0 0 2 2 2 1 1 1 2 2 2");
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  arma::vec classProbabilities;
  AllCategoricalSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 3, weights);
  const double gain = AllCategoricalSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, 4, labels, 3, weights, 3, 1e-7, classProbabilities,
      aux);
  const double weightedGain =
      AllCategoricalSplit<GiniGain>::SplitIfBetter<true>(bestGain, values, 4,
      labels, 3, weights, 3, 1e-7, classProbabilities, aux);

  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  // Since the split is perfect, make sure the new gain is 0.
  REQUIRE(gain == Approx(0.0).margin(1e-7));

  REQUIRE(gain == weightedGain);

  // Make sure the class probabilities now hold the number of children.
  REQUIRE(classProbabilities.n_elem == 1);
  REQUIRE((size_t) classProbabilities[0] == 4);
}

/**
 * Make sure that AllCategoricalSplit respects the minimum number of samples
 * required to split.
 */
TEST_CASE("AllCategoricalSplitMinSamplesTest", "[DecisionTreeTest]")
{
  arma::vec values("0 0 0 1 1 1 2 2 2 3 3 3");
  arma::Row<size_t> labels("0 0 0 2 2 2 1 1 1 2 2 2");
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  arma::vec classProbabilities;
  AllCategoricalSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 3, weights);
  const double gain = AllCategoricalSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, 4, labels, 3, weights, 4, 1e-7,
      classProbabilities, aux);

  // Make sure it's not split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(classProbabilities.n_elem == 0);
}

/**
 * Check that no split is made when it doesn't get us anything.
 */
TEST_CASE("AllCategoricalSplitNoGainTest", "[DecisionTreeTest]")
{
  arma::vec values(300);
  arma::Row<size_t> labels(300);
  arma::rowvec weights = arma::ones<arma::rowvec>(300);

  for (size_t i = 0; i < 300; i += 3)
  {
    values[i] = int(i / 3) % 10;
    labels[i] = 0;
    values[i + 1] = int(i / 3) % 10;
    labels[i + 1] = 1;
    values[i + 2] = int(i / 3) % 10;
    labels[i + 2] = 2;
  }

  arma::vec classProbabilities;
  AllCategoricalSplit<GiniGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = GiniGain::Evaluate<false>(labels, 3, weights);
  const double gain = AllCategoricalSplit<GiniGain>::SplitIfBetter<false>(
      bestGain, values, 10, labels, 3, weights, 10, 1e-7,
      classProbabilities, aux);
  const double weightedGain =
      AllCategoricalSplit<GiniGain>::SplitIfBetter<true>(bestGain, values, 10,
      labels, 3, weights, 10, 1e-7, classProbabilities, aux);

  // Make sure that there was no split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
  REQUIRE(classProbabilities.n_elem == 0);
}

/**
 * A basic construction of the decision tree---ensure that we can create the
 * tree and that it split at least once.
 */
TEST_CASE("BasicConstructionTest", "[DecisionTreeTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = 0.0;
    labels[i] = 0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = 1.0;
    labels[i] = 1;
  }

  // Use default parameters.
  DecisionTree<> d(dataset, labels, 2, 10);

  // Now require that we have some children.
  REQUIRE(d.NumChildren() > 0);
}

/**
 * Construct a tree with weighted labels.
 */
TEST_CASE("BasicConstructionTestWithWeight", "[DecisionTreeTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = 0.0;
    labels[i] = 0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = 1.0;
    labels[i] = 1;
  }
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Use default parameters.
  DecisionTree<> wd(dataset, labels, 2, weights, 10);
  DecisionTree<> d(dataset, labels, 2, 10);

  // Now require that we have some children.
  REQUIRE(wd.NumChildren() > 0);
  REQUIRE(wd.NumChildren() == d.NumChildren());
}

/**
 * Construct the decision tree on numeric data only and see that we can fit it
 * exactly and achieve perfect performance on the training set.
 */
TEST_CASE("PerfectTrainingSet", "[DecisionTreeTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = 0.0;
    labels[i] = 0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = 1.0;
    labels[i] = 1;
  }

  DecisionTree<> d(dataset, labels, 2, 1, 0.0); // Minimum leaf size of 1.

  // Make sure that we can get perfect accuracy on the training set.
  for (size_t i = 0; i < 100; ++i)
  {
    size_t prediction;
    arma::vec probabilities;
    d.Classify(dataset.col(i), prediction, probabilities);

    REQUIRE(prediction == labels[i]);
    REQUIRE(probabilities.n_elem == 2);
    for (size_t j = 0; j < 2; ++j)
    {
      if (labels[i] == j)
        REQUIRE(probabilities[j] == Approx(1.0).epsilon(1e-7));
      else
        REQUIRE(probabilities[j] == Approx(0.0).margin(1e-5));
    }
  }
}

/**
 * Construct the decision tree with weighted labels
 */
TEST_CASE("PerfectTrainingSetWithWeight", "[DecisionTreeTest]")
{
  // Completely random dataset with no structure.
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = 0.0;
    labels[i] = 0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = 1.0;
    labels[i] = 1;
  }
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTree<> d(dataset, labels, 2, weights, 1, 0.0);

  // This part of code is dupliacte with no weighted one.
  for (size_t i = 0; i < 100; ++i)
  {
    size_t prediction;
    arma::vec probabilities;
    d.Classify(dataset.col(i), prediction, probabilities);

    REQUIRE(prediction == labels[i]);
    REQUIRE(probabilities.n_elem == 2);
    for (size_t j = 0; j < 2; ++j)
    {
      if (labels[i] == j)
        REQUIRE(probabilities[j] == Approx(1.0).epsilon(1e-7));
      else
        REQUIRE(probabilities[j] == Approx(0.0).margin(1e-5));
    }
  }
}


/**
 * Make sure class probabilities are computed correctly in the root node.
 */
TEST_CASE("ClassProbabilityTest", "[DecisionTreeTest]")
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

  REQUIRE(d.NumChildren() == 0);

  // Estimate a point's probabilities.
  arma::vec probabilities;
  size_t prediction;
  d.Classify(dataset.col(0), prediction, probabilities);

  REQUIRE(probabilities.n_elem == 2);
  REQUIRE(probabilities[0] == Approx(0.5).epsilon(1e-7));
  REQUIRE(probabilities[1] == Approx(0.5).epsilon(1e-7));
}

/**
 * Test that the decision tree generalizes reasonably.
 */
TEST_CASE("SimpleGeneralizationTest", "[DecisionTreeTest]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::rowvec weights(labels.n_cols, arma::fill::ones);

  // Build decision tree.
  DecisionTree<> d(inputData, labels, 3, 10); // Leaf size of 10.
  DecisionTree<> wd(inputData, labels, 3, weights, 10); // Leaf size of 10.

  // Load testing data.
  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");

  arma::Mat<size_t> trueTestLabels;
  if (!data::Load("vc2_test_labels.txt", trueTestLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt");

  // Get the predicted test labels.
  arma::Row<size_t> predictions;
  d.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the accuracy.
  double correct = 0.0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions[i] == trueTestLabels[i])
      ++correct;
  correct /= predictions.n_elem;

  REQUIRE(correct > 0.75);

  // reset the prediction
  predictions.zeros();
  wd.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the accuracy.
  double wdcorrect = 0.0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions[i] == trueTestLabels[i])
      ++wdcorrect;
  wdcorrect /= predictions.n_elem;

  REQUIRE(wdcorrect > 0.75);
}

/**
 * Test that the decision tree generalizes reasonably when built on float data.
 */
TEST_CASE("SimpleGeneralizationFMatTest", "[DecisionTreeTest]")
{
  arma::fmat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::rowvec weights(labels.n_cols, arma::fill::ones);

  // Build decision tree.
  DecisionTree<> d(inputData, labels, 3, 10 /* Leaf size of 10. */);
  DecisionTree<> wd(inputData, labels, 3, weights, 10 /* Leaf size of 10. */);

  // Load testing data.
  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");

  arma::Mat<size_t> trueTestLabels;
  if (!data::Load("vc2_test_labels.txt", trueTestLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt");

  // Get the predicted test labels.
  arma::Row<size_t> predictions;
  d.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the accuracy.
  double correct = 0.0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions[i] == trueTestLabels[i])
      ++correct;
  correct /= predictions.n_elem;

  REQUIRE(correct > 0.75);

  // Reset the prediction.
  predictions.zeros();
  wd.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the accuracy.
  double wdcorrect = 0.0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions[i] == trueTestLabels[i])
      ++wdcorrect;
  wdcorrect /= predictions.n_elem;

  REQUIRE(wdcorrect > 0.75);
}

/**
 * Test that we can build a decision tree on a simple categorical dataset.
 */
TEST_CASE("CategoricalBuildTest", "[DecisionTreeTest]")
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

  // Build the tree.
  DecisionTree<> tree(trainingData, di, trainingLabels, 5, 10);

  // Now evaluate the accuracy of the tree.
  arma::Row<size_t> predictions;
  tree.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);
  size_t correct = 0;
  for (size_t i = 0; i < testData.n_cols; ++i)
    if (testLabels[i] == predictions[i])
      ++correct;

  // Make sure we got at least 70% accuracy.
  const double correctPct = double(correct) / double(testData.n_cols);
  REQUIRE(correctPct > 0.70);
}

/**
 * Test that we can build a decision tree with weights on a simple categorical
 * dataset.
 */
TEST_CASE("CategoricalBuildTestWithWeight", "[DecisionTreeTest]")
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

  arma::Row<double> weights = arma::ones<arma::Row<double>>(
      trainingLabels.n_elem);

  // Build the tree.
  DecisionTree<> tree(trainingData, di, trainingLabels, 5, weights, 10);

  // Now evaluate the accuracy of the tree.
  arma::Row<size_t> predictions;
  tree.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);
  size_t correct = 0;
  for (size_t i = 0; i < testData.n_cols; ++i)
    if (testLabels[i] == predictions[i])
      ++correct;

  // Make sure we got at least 70% accuracy.
  const double correctPct = double(correct) / double(testData.n_cols);
  REQUIRE(correctPct > 0.70);
}

/**
 * Test that we can build a decision tree using weighted data (where the
 * low-weighted data is random noise), and that the tree still builds correctly
 * enough to get good results.
 */
TEST_CASE("WeightedDecisionTreeTest", "[DecisionTreeTest]")
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load test dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt!");

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

  // Now build the decision tree.  I think the syntax is right here.
  DecisionTree<> d(data, fullLabels, 3, weights, 10);

  // Now we can check that we get good performance on the VC2 test set.
  arma::mat testData;
  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");
  if (!data::Load("vc2_test_labels.txt", testLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt!");

  arma::Row<size_t> predictions;
  d.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the accuracy.
  double correct = 0.0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions[i] == testLabels[i])
      ++correct;
  correct /= predictions.n_elem;

  REQUIRE(correct > 0.75);
}
/**
 * Test that we can build a decision tree on a simple categorical dataset using
 * weights, with low-weight noise added.
 */
TEST_CASE("CategoricalWeightedBuildTest", "[DecisionTreeTest]")
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

  // Build the tree.
  DecisionTree<> tree(fullData, di, fullLabels, 5, weights, 10);

  // Now evaluate the accuracy of the tree.
  arma::Row<size_t> predictions;
  tree.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);
  size_t correct = 0;
  for (size_t i = 0; i < testData.n_cols; ++i)
    if (testLabels[i] == predictions[i])
      ++correct;

  // Make sure we got at least 70% accuracy.
  const double correctPct = double(correct) / double(testData.n_cols);
  REQUIRE(correctPct > 0.70);
}

/**
 * Test that we can build a decision tree using weighted data (where the
 * low-weighted data is random noise) with information gain, and that the tree
 * still builds correctly enough to get good results.
 */
TEST_CASE("WeightedDecisionTreeInformationGainTest", "[DecisionTreeTest]")
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load test dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt!");

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

  // Now build the decision tree.  I think the syntax is right here.
  DecisionTree<InformationGain> d(data, fullLabels, 3, weights, 10);

  // Now we can check that we get good performance on the VC2 test set.
  arma::mat testData;
  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");
  if (!data::Load("vc2_test_labels.txt", testLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt!");

  arma::Row<size_t> predictions;
  d.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the accuracy.
  double correct = 0.0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions[i] == testLabels[i])
      ++correct;
  correct /= predictions.n_elem;

  REQUIRE(correct > 0.75);
}
/**
 * Test that we can build a decision tree using information gain on a simple
 * categorical dataset using weights, with low-weight noise added.
 */
TEST_CASE("CategoricalInformationGainWeightedBuildTest", "[DecisionTreeTest]")
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

  // Build the tree.
  DecisionTree<InformationGain> tree(fullData, di, fullLabels, 5, weights, 10);

  // Now evaluate the accuracy of the tree.
  arma::Row<size_t> predictions;
  tree.Classify(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);
  size_t correct = 0;
  for (size_t i = 0; i < testData.n_cols; ++i)
    if (testLabels[i] == predictions[i])
      ++correct;

  // Make sure we got at least 70% accuracy.
  const double correctPct = double(correct) / double(testData.n_cols);
  REQUIRE(correctPct > 0.70);
}

/**
 * Make sure that the random dimension selector only has one element.
 */
TEST_CASE("RandomDimensionSelectTest", "[DecisionTreeTest]")
{
  RandomDimensionSelect r;
  r.Dimensions() = 10;

  REQUIRE(r.Begin() < 10);
  REQUIRE(r.Next() == r.End());
  REQUIRE(r.Next() == r.End());
  REQUIRE(r.Next() == r.End());
}

/**
 * Make sure that the random dimension selector selects different values.
 */
TEST_CASE("RandomDimensionSelectRandomTest", "[DecisionTreeTest]")
{
  // We'll check that 4 values are not all the same.
  RandomDimensionSelect r1, r2, r3, r4;
  r1.Dimensions() = 100000;
  r2.Dimensions() = 100000;
  r3.Dimensions() = 100000;
  r4.Dimensions() = 100000;

  REQUIRE(((r1.Begin() != r2.Begin()) ||
           (r1.Begin() != r3.Begin()) ||
           (r1.Begin() != r4.Begin())));
}

/**
 * Make sure that the multiple random dimension select only has the right number
 * of elements.
 */
TEST_CASE("MultipleRandomDimensionSelectTest", "[DecisionTreeTest]")
{
  MultipleRandomDimensionSelect r(5);
  r.Dimensions() = 10;

  // Make sure we get five elements.
  REQUIRE(r.Begin() < 10);
  REQUIRE(r.Next() < 10);
  REQUIRE(r.Next() < 10);
  REQUIRE(r.Next() < 10);
  REQUIRE(r.Next() < 10);
  REQUIRE(r.Next() == r.End());
}

/**
 * Make sure we get every element from the distribution.
 */
TEST_CASE("MultipleRandomDimensionAllSelectTest", "[DecisionTreeTest]")
{
  MultipleRandomDimensionSelect r(3);
  r.Dimensions() = 3;

  bool found[3];
  found[0] = found[1] = found[2] = false;

  found[r.Begin()] = true;
  found[r.Next()] = true;
  found[r.Next()] = true;

  REQUIRE(found[0] == true);
  REQUIRE(found[1] == true);
  REQUIRE(found[2] == true);
}

/**
 * Make sure the right number of classes is returned for an empty tree (1).
 */
TEST_CASE("NumClassesEmptyTreeTest", "[DecisionTreeTest]")
{
  DecisionTree<> dt;
  REQUIRE(dt.NumClasses() == 1);
}

/**
 * Make sure the right number of classes is returned for a nonempty tree.
 */
TEST_CASE("NumClassesTest", "[DecisionTreeTest]")
{
  // Load a dataset to train with.
  arma::mat dataset;
  arma::Row<size_t> labels;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load test dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt!");

  DecisionTree<> dt(dataset, labels, 3);

  REQUIRE(dt.NumClasses() == 3);
}

/*
 * Test that we can pass const data into DecisionTree constructors.
 */
TEST_CASE("ConstDataTest", "[DecisionTreeTest]")
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);

  const arma::mat& constData = data;
  const arma::Row<size_t>& constLabels = labels;
  const arma::rowvec constWeights(labels.n_elem, arma::fill::randu);
  const size_t numClasses = 5;

  DecisionTree<> dt(constData, constLabels, numClasses);
  DecisionTree<> dt2(constData, datasetInfo, constLabels, numClasses);
  DecisionTree<> dt3(constData, constLabels, numClasses, constWeights);
  DecisionTree<> dt4(constData, datasetInfo, constLabels, numClasses,
      constWeights);
}

/**
 * Construct the decision tree with splitting only if gain is more than
 * threshold.
 */
TEST_CASE("RegularisedDecisionTree", "[DecisionTreeTest]")
{
  // Completely random dataset with no structure.
  arma::mat dataset(10, 1000, arma::fill::randu);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 1000; ++i)
    labels[i] = i % 3; // 3 classes.
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTree<> d(dataset, labels, 3, weights, 1, 1e-7);

  // Minimum leaf size of 1 and Minimum gain split of 0.01.
  DecisionTree<> dRegularised(dataset, labels, 3, weights, 1, 0.01);

  size_t count = 0;
  // This part of code is dupliacte with no weighted one.
  for (size_t i = 0; i < 1000; ++i)
  {
    size_t prediction, predictionsregularised;
    arma::vec probabilities, probabilitiesRegularised;

    d.Classify(dataset.col(i), prediction, probabilities);
    dRegularised.Classify(dataset.col(i), predictionsregularised,
                          probabilitiesRegularised);

    if (prediction != predictionsregularised)
      count++;

    REQUIRE(probabilities.n_elem == 3);
    REQUIRE(probabilitiesRegularised.n_elem == 3);
  }

  REQUIRE(count > 0);
}

/**
 * Test that DecisionTree::Train() returns finite entropy on numeric dataset.
 */
TEST_CASE("DecisionTreeNumericTrainReturnEntropy", "[DecisionTreeTest]")
{
  arma::mat dataset(10, 1000, arma::fill::randu);
  arma::Row<size_t> labels(1000);
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  for (size_t i = 0; i < 1000; ++i)
    labels[i] = i % 3; // 3 classes.

  // Train a simpe tree on numeric dataset.
  DecisionTree<> d(3);
  double entropy = d.Train(dataset, labels, 3, 50);

  REQUIRE(std::isfinite(entropy) == true);

  // Train a tree with weights on numeric dataset.
  DecisionTree<> wd(3);
  entropy = wd.Train(dataset, labels, 3, weights, 50);

  REQUIRE(std::isfinite(entropy) == true);
}

/**
 * Test that DecisionTree::Train() returns finite entropy on categorical
 * dataset.
 */
TEST_CASE("DecisionTreeCategoricalTrainReturnEntropy", "[DecisionTreeTest]")
{
  arma::mat d;
  arma::Row<size_t> l;
  data::DatasetInfo di;
  MockCategoricalData(d, l, di);

  arma::Row<double> weights = arma::ones<arma::Row<double>>(l.n_elem);

  // Train a simple tree on categorical dataset.
  DecisionTree<> dtree(5);
  double entropy = dtree.Train(d, di, l, 5, 10);

  REQUIRE(std::isfinite(entropy) == true);

  // Train a tree with weights on categorical dataset.
  DecisionTree<> wdtree(5);
  entropy = wdtree.Train(d, di, l, 5, weights, 10);

  REQUIRE(std::isfinite(entropy) == true);
}

/**
 * Make sure different maximum depth values give different numbers of children.
 */
TEST_CASE("DifferentMaximumDepthTest", "[DecisionTreeTest]")
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  if (!data::Load("vc2.csv", dataset))
    FAIL("Cannot load test dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt!");

  DecisionTree<> d(dataset, labels, 3, 10, 1e-7, 1);

  DecisionTree<> d1(dataset, labels, 3, 10, 1e-7, 2);

  DecisionTree<> d2(dataset, labels, 3, 10, 1e-7);

  // Now require that we have zero children.
  REQUIRE(d.NumChildren() == 0);

  // Now require that we have two children.
  REQUIRE(d1.NumChildren() == 2);
  REQUIRE(d1.Child(0).NumChildren() == 0);
  REQUIRE(d1.Child(1).NumChildren() == 0);

  // Now require that we have two children.
  REQUIRE(d2.NumChildren() == 2);
  REQUIRE(d2.Child(0).NumChildren() == 2);
  REQUIRE(d2.Child(1).NumChildren() == 2);
}
