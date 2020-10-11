/**
 * @file tests/hoeffding_tree_test.cpp
 * @author Ryan Curtin
 *
 * Test file for Hoeffding trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/gini_impurity.hpp>
#include <mlpack/methods/hoeffding_trees/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_categorical_split.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_model.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization_catch.hpp"

#include <stack>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::data;
using namespace mlpack::tree;

TEST_CASE("GiniImpurityPerfectSimpleTest", "[HoeffdingTreeTest]")
{
  // Make a simple test for Gini impurity with one class.  In this case it
  // should always be 0.  We'll assemble the count matrix by hand.
  arma::Mat<size_t> counts(2, 2); // 2 categories, 2 classes.

  counts(0, 0) = 10; // 10 points in category 0 with class 0.
  counts(0, 1) = 0; // 0 points in category 0 with class 1.
  counts(1, 0) = 12; // 12 points in category 1 with class 0.
  counts(1, 1) = 0; // 0 points in category 1 with class 1.

  // Since the split gets us nothing, there should be no gain.
  REQUIRE(GiniImpurity::Evaluate(counts) == Approx(0.0).margin(1e-10));
}

TEST_CASE("GiniImpurityImperfectSimpleTest", "[HoeffdingTreeTest]")
{
  // Make a simple test where a split will give us perfect classification.
  arma::Mat<size_t> counts(2, 2); // 2 categories, 2 classes.

  counts(0, 0) = 10; // 10 points in category 0 with class 0.
  counts(1, 0) = 0; // 0 points in category 0 with class 1.
  counts(0, 1) = 0; // 0 points in category 1 with class 0.
  counts(1, 1) = 10; // 10 points in category 1 with class 1.

  // The impurity before the split should be 0.5^2 + 0.5^2 = 0.5.
  // The impurity after the split should be 0.
  // So the gain should be 0.5.
  REQUIRE(GiniImpurity::Evaluate(counts) == Approx(0.5).epsilon(1e-7));
}

TEST_CASE("GiniImpurityBadSplitTest", "[HoeffdingTreeTest]")
{
  // Make a simple test where a split gets us nothing.
  arma::Mat<size_t> counts(2, 2);
  counts(0, 0) = 10;
  counts(0, 1) = 10;
  counts(1, 0) = 5;
  counts(1, 1) = 5;

  REQUIRE(GiniImpurity::Evaluate(counts) == Approx(0.0).margin(1e-10));
}

/**
 * A hand-crafted more difficult test for the Gini impurity, where four
 * categories and three classes are available.
 */
TEST_CASE("GiniImpurityThreeClassTest", "[HoeffdingTreeTest]")
{
  arma::Mat<size_t> counts(3, 4);

  counts(0, 0) = 0;
  counts(1, 0) = 0;
  counts(2, 0) = 10;

  counts(0, 1) = 5;
  counts(1, 1) = 5;
  counts(2, 1) = 0;

  counts(0, 2) = 4;
  counts(1, 2) = 4;
  counts(2, 2) = 4;

  counts(0, 3) = 8;
  counts(1, 3) = 1;
  counts(2, 3) = 1;

  // The Gini impurity of the whole thing is:
  // (overall sum) 0.65193 -
  // (category 0)  0.40476 * 0       -
  // (category 1)  0.23810 * 0.5     -
  // (category 2)  0.28571 * 0.66667 -
  // (category 2)  0.23810 * 0.34
  //   = 0.26145
  REQUIRE(GiniImpurity::Evaluate(counts) == Approx(0.26145).epsilon(1e-5));
}

TEST_CASE("GiniImpurityZeroTest", "[HoeffdingTreeTest]")
{
  // When nothing has been seen, the gini impurity should be zero.
  arma::Mat<size_t> counts = arma::zeros<arma::Mat<size_t>>(10, 10);

  REQUIRE(GiniImpurity::Evaluate(counts) == Approx(0.0).margin(1e-10));
}

/**
 * Test that the range of Gini impurities is correct for a handful of class
 * sizes.
 */
TEST_CASE("GiniImpurityRangeTest", "[HoeffdingTreeTest]")
{
  REQUIRE(GiniImpurity::Range(1) == Approx(0).epsilon(1e-7));
  REQUIRE(GiniImpurity::Range(2) == Approx(0.5).epsilon(1e-7));
  REQUIRE(GiniImpurity::Range(3) == Approx(0.66666667).epsilon(1e-7));
  REQUIRE(GiniImpurity::Range(4) == Approx(0.75).epsilon(1e-7));
  REQUIRE(GiniImpurity::Range(5) == Approx(0.8).epsilon(1e-7));
  REQUIRE(GiniImpurity::Range(10) == Approx(0.9).epsilon(1e-7));
  REQUIRE(GiniImpurity::Range(100) == Approx(0.99).epsilon(1e-7));
  REQUIRE(GiniImpurity::Range(1000) == Approx(0.999).epsilon(1e-7));
}

TEST_CASE("HoeffdingInformationGainPerfectSimpleTest", "[HoeffdingTreeTest]")
{
  // Make a simple test for Gini impurity with one class.  In this case it
  // should always be 0.  We'll assemble the count matrix by hand.
  arma::Mat<size_t> counts(2, 2); // 2 categories, 2 classes.

  counts(0, 0) = 10; // 10 points in category 0 with class 0.
  counts(0, 1) = 0; // 0 points in category 0 with class 1.
  counts(1, 0) = 12; // 12 points in category 1 with class 0.
  counts(1, 1) = 0; // 0 points in category 1 with class 1.

  // Since the split gets us nothing, there should be no gain.
  REQUIRE(HoeffdingInformationGain::Evaluate(counts) ==
      Approx(0.0).margin(1e-10));
}

TEST_CASE("HoeffdingInformationGainImperfectSimpleTest", "[HoeffdingTreeTest]")
{
  // Make a simple test where a split will give us perfect classification.
  arma::Mat<size_t> counts(2, 2); // 2 categories, 2 classes.

  counts(0, 0) = 10; // 10 points in category 0 with class 0.
  counts(1, 0) = 0; // 0 points in category 0 with class 1.
  counts(0, 1) = 0; // 0 points in category 1 with class 0.
  counts(1, 1) = 10; // 10 points in category 1 with class 1.

  // The impurity before the split should be 0.5 log2(0.5) + 0.5 log2(0.5) = -1.
  // The impurity after the split should be 0.
  // So the gain should be 1.
  REQUIRE(HoeffdingInformationGain::Evaluate(counts) ==
      Approx(1.0).epsilon(1e-7));
}

TEST_CASE("HoeffdingInformationGainBadSplitTest", "[HoeffdingTreeTest]")
{
  // Make a simple test where a split gets us nothing.
  arma::Mat<size_t> counts(2, 2);
  counts(0, 0) = 10;
  counts(0, 1) = 10;
  counts(1, 0) = 5;
  counts(1, 1) = 5;

  REQUIRE(HoeffdingInformationGain::Evaluate(counts) == Approx(0.0).margin(1e-10));
}

/**
 * A hand-crafted more difficult test for the Gini impurity, where four
 * categories and three classes are available.
 */
TEST_CASE("HoeffdingInformationGainThreeClassTest", "[HoeffdingTreeTest]")
{
  arma::Mat<size_t> counts(3, 4);

  counts(0, 0) = 0;
  counts(1, 0) = 0;
  counts(2, 0) = 10;

  counts(0, 1) = 5;
  counts(1, 1) = 5;
  counts(2, 1) = 0;

  counts(0, 2) = 4;
  counts(1, 2) = 4;
  counts(2, 2) = 4;

  counts(0, 3) = 8;
  counts(1, 3) = 1;
  counts(2, 3) = 1;

  // The Gini impurity of the whole thing is:
  // (overall sum) -1.5516 +
  // (category 0)  0.40476 * 0       -
  // (category 1)  0.23810 * -1      -
  // (category 2)  0.28571 * -1.5850 -
  // (category 3)  0.23810 * -0.92193
  //   = 0.64116649
  REQUIRE(HoeffdingInformationGain::Evaluate(counts) ==
      Approx(0.64116649).epsilon(1e-7));
}

TEST_CASE("HoeffdingInformationGainZeroTest", "[HoeffdingTreeTest]")
{
  // When nothing has been seen, the information gain should be zero.
  arma::Mat<size_t> counts = arma::zeros<arma::Mat<size_t>>(10, 10);

  REQUIRE(HoeffdingInformationGain::Evaluate(counts) == Approx(0.0).margin(1e-10));
}

/**
 * Test that the range of information gains is correct for a handful of class
 * sizes.
 */
TEST_CASE("HoeffdingInformationGainRangeTest", "[HoeffdingTreeTest]")
{
  REQUIRE(HoeffdingInformationGain::Range(1) == Approx(0).epsilon(1e-7));
  REQUIRE(HoeffdingInformationGain::Range(2) == Approx(1.0).epsilon(1e-7));
  REQUIRE(HoeffdingInformationGain::Range(3) == Approx(1.5849625).epsilon(1e-7));
  REQUIRE(HoeffdingInformationGain::Range(4) == Approx(2).epsilon(1e-7));
  REQUIRE(HoeffdingInformationGain::Range(5) == Approx(2.32192809).epsilon(1e-7));
  REQUIRE(HoeffdingInformationGain::Range(10) == Approx(3.32192809).epsilon(1e-7));
  REQUIRE(HoeffdingInformationGain::Range(100) == Approx(6.64385619).epsilon(1e-7));
  REQUIRE(HoeffdingInformationGain::Range(1000) == Approx(9.96578428).epsilon(1e-7));
}

/**
 * Feed the HoeffdingCategoricalSplit class many examples, all from the same
 * class, and verify that the majority class is correct.
 */
TEST_CASE("HoeffdingCategoricalSplitMajorityClassTest", "[HoeffdingTreeTest]")
{
  // Ten categories, three classes.
  HoeffdingCategoricalSplit<GiniImpurity> split(10, 3);

  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(mlpack::math::RandInt(0, 10), 1);
    REQUIRE(split.MajorityClass() == 1);
  }
}

/**
 * A harder majority class example.
 */
TEST_CASE("HoeffdingCategoricalSplitHarderMajorityClassTest",
          "[HoeffdingTreeTest]")
{
  // Ten categories, three classes.
  HoeffdingCategoricalSplit<GiniImpurity> split(10, 3);

  split.Train(mlpack::math::RandInt(0, 10), 1);
  for (size_t i = 0; i < 250; ++i)
  {
    split.Train(mlpack::math::RandInt(0, 10), 1);
    split.Train(mlpack::math::RandInt(0, 10), 2);
    REQUIRE(split.MajorityClass() == 1);
  }
}

/**
 * Ensure that the fitness function is positive when we pass some data that
 * would result in an improvement if it was split.
 */
TEST_CASE("HoeffdingCategoricalSplitEasyFitnessCheck", "[HoeffdingTreeTest]")
{
  HoeffdingCategoricalSplit<GiniImpurity> split(5, 3);

  for (size_t i = 0; i < 100; ++i)
    split.Train(0, 0);
  for (size_t i = 0; i < 100; ++i)
    split.Train(1, 1);
  for (size_t i = 0; i < 100; ++i)
    split.Train(2, 1);
  for (size_t i = 0; i < 100; ++i)
    split.Train(3, 2);
  for (size_t i = 0; i < 100; ++i)
    split.Train(4, 2);

  double bestGain, secondBestGain;
  split.EvaluateFitnessFunction(bestGain, secondBestGain);
  REQUIRE(bestGain > 0.0);
  REQUIRE(secondBestGain == Approx(0.0).margin(1e-10));
}

/**
 * Ensure that the fitness function returns 0 (no improvement) when a split
 * would not get us any improvement.
 */
TEST_CASE("HoeffdingCategoricalSplitNoImprovementFitnessTest",
          "[HoeffdingTreeTest]")
{
  HoeffdingCategoricalSplit<GiniImpurity> split(2, 2);

  // No training has yet happened, so a split would get us nothing.
  double bestGain, secondBestGain;
  split.EvaluateFitnessFunction(bestGain, secondBestGain);
  REQUIRE(bestGain == Approx(0.0).margin(1e-10));
  REQUIRE(secondBestGain == Approx(0.0).margin(1e-10));

  split.Train(0, 0);
  split.Train(1, 0);
  split.Train(0, 1);
  split.Train(1, 1);

  // Now, a split still gets us only 50% accuracy in each split bin.
  split.EvaluateFitnessFunction(bestGain, secondBestGain);
  REQUIRE(bestGain == Approx(0.0).margin(1e-10));
  REQUIRE(secondBestGain == Approx(0.0).margin(1e-10));
}

/**
 * Test that when we do split, we get reasonable split information.
 */
TEST_CASE("HoeffdingCategoricalSplitSplitTest", "[HoeffdingTreeTest]")
{
  HoeffdingCategoricalSplit<GiniImpurity> split(3, 3); // 3 categories.

  // No training is necessary because we can just call CreateChildren().
  data::DatasetInfo info(3);
  info.MapString<size_t>("hello", 0); // Make dimension 0 categorical.
  HoeffdingCategoricalSplit<GiniImpurity>::SplitInfo splitInfo(3);

  // Create the children.
  arma::Col<size_t> childMajorities;
  split.Split(childMajorities, splitInfo);

  REQUIRE(childMajorities.n_elem == 3);
  REQUIRE(splitInfo.CalculateDirection(0) == 0);
  REQUIRE(splitInfo.CalculateDirection(1) == 1);
  REQUIRE(splitInfo.CalculateDirection(2) == 2);
}

/**
 * If we feed the HoeffdingTree a ton of points of the same class, it should
 * not suggest that we split.
 */
TEST_CASE("HoeffdingTreeNoSplitTest", "[HoeffdingTreeTest]")
{
  // Make all dimensions categorical.
  data::DatasetInfo info(3);
  info.MapString<size_t>("cat1", 0);
  info.MapString<size_t>("cat2", 0);
  info.MapString<size_t>("cat3", 0);
  info.MapString<size_t>("cat4", 0);
  info.MapString<size_t>("cat1", 1);
  info.MapString<size_t>("cat2", 1);
  info.MapString<size_t>("cat3", 1);
  info.MapString<size_t>("cat1", 2);
  info.MapString<size_t>("cat2", 2);

  HoeffdingTree<> split(info, 2, 0.95, 5000, 1);

  // Feed it samples.
  for (size_t i = 0; i < 1000; ++i)
  {
    // Create the test point.
    arma::Col<size_t> testPoint(3);
    testPoint(0) = mlpack::math::RandInt(0, 4);
    testPoint(1) = mlpack::math::RandInt(0, 3);
    testPoint(2) = mlpack::math::RandInt(0, 2);
    split.Train(testPoint, 0); // Always label 0.

    REQUIRE(split.SplitCheck() == 0);
  }
}

/**
 * If we feed the HoeffdingTree a ton of points of two different classes, it
 * should very clearly suggest that we split (eventually).
 */
TEST_CASE("HoeffdingTreeEasySplitTest", "[HoeffdingTreeTest]")
{
  // It'll be a two-dimensional dataset with two categories each.  In the first
  // dimension, category 0 will only receive points with class 0, and category 1
  // will only receive points with class 1.  In the second dimension, all points
  // will have category 0 (so it is useless).
  data::DatasetInfo info(2);
  info.MapString<size_t>("cat0", 0);
  info.MapString<size_t>("cat1", 0);
  info.MapString<size_t>("cat0", 1);

  HoeffdingTree<> tree(info, 2, 0.95, 5000, 5000 /* never check for splits */);

  // Feed samples from each class.
  for (size_t i = 0; i < 500; ++i)
  {
    tree.Train(arma::Col<size_t>("0 0"), 0);
    tree.Train(arma::Col<size_t>("1 0"), 1);
  }

  // Now it should be ready to split.
  REQUIRE(tree.SplitCheck() == 2);
  REQUIRE(tree.SplitDimension() == 0);
}

/**
 * If we force a success probability of 1, it should never split.
 */
TEST_CASE("HoeffdingTreeProbability1SplitTest", "[HoeffdingTreeTest]")
{
  // It'll be a two-dimensional dataset with two categories each.  In the first
  // dimension, category 0 will only receive points with class 0, and category 1
  // will only receive points with class 1.  In the second dimension, all points
  // will have category 0 (so it is useless).
  data::DatasetInfo info(2);
  info.MapString<size_t>("cat0", 0);
  info.MapString<size_t>("cat1", 0);
  info.MapString<size_t>("cat0", 1);

  HoeffdingTree<> split(info, 2, 1.0, 12000, 1 /* always check for splits */);

  // Feed samples from each class.
  for (size_t i = 0; i < 5000; ++i)
  {
    split.Train(arma::Col<size_t>("0 0"), 0);
    split.Train(arma::Col<size_t>("1 0"), 1);
  }

  // But because the success probability is 1, it should never split.
  REQUIRE(split.SplitCheck() == 0);
  REQUIRE(split.SplitDimension() == size_t(-1));
}

/**
 * A slightly harder splitting problem: there are two features; one gives
 * perfect classification, another gives almost perfect classification (with 10%
 * error).  Splits should occur after many samples.
 */
TEST_CASE("HoeffdingTreeAlmostPerfectSplit", "[HoeffdingTreeTest]")
{
  // Two categories and two dimensions.
  data::DatasetInfo info(2);
  info.MapString<size_t>("cat0", 0);
  info.MapString<size_t>("cat1", 0);
  info.MapString<size_t>("cat0", 1);
  info.MapString<size_t>("cat1", 1);

  HoeffdingTree<> split(info, 2, 0.95, 5000, 5000 /* never check for splits */);

  // Feed samples.
  for (size_t i = 0; i < 500; ++i)
  {
    if (mlpack::math::Random() <= 0.9)
      split.Train(arma::Col<size_t>("0 0"), 0);
    else
      split.Train(arma::Col<size_t>("1 0"), 0);

    if (mlpack::math::Random() <= 0.9)
      split.Train(arma::Col<size_t>("1 1"), 1);
    else
      split.Train(arma::Col<size_t>("0 1"), 1);
  }

  // Ensure that splitting should happen.
  REQUIRE(split.SplitCheck() == 2);
  // Make sure that it's split on the correct dimension.
  REQUIRE(split.SplitDimension() == 1);
}

/**
 * Test that the HoeffdingTree class will not split if the two features are
 * equally good.
 */
TEST_CASE("HoeffdingTreeEqualSplitTest", "[HoeffdingTreeTest]")
{
  // Two categories and two dimensions.
  data::DatasetInfo info(2);
  info.MapString<size_t>("cat0", 0);
  info.MapString<size_t>("cat1", 0);
  info.MapString<size_t>("cat0", 1);
  info.MapString<size_t>("cat1", 1);

  HoeffdingTree<> split(info, 2, 0.95, 5000, 1);

  // Feed samples.
  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(arma::Col<size_t>("0 0"), 0);
    split.Train(arma::Col<size_t>("1 1"), 1);
  }

  // Ensure that splitting should not happen.
  REQUIRE(split.SplitCheck() == 0);
}

// This is used in the next test.
template<typename FitnessFunction>
using HoeffdingSizeTNumericSplit = HoeffdingNumericSplit<FitnessFunction,
    size_t>;

/**
 * Build a decision tree on a dataset with two meaningless dimensions and ensure
 * that it can properly classify all of the training points.  (The dataset is
 * perfectly separable.)
 */
TEST_CASE("HoeffdingTreeSimpleDatasetTest", "[HoeffdingTreeTest]")
{
  DatasetInfo info(3);
  info.MapString<size_t>("cat0", 0);
  info.MapString<size_t>("cat1", 0);
  info.MapString<size_t>("cat2", 0);
  info.MapString<size_t>("cat3", 0);
  info.MapString<size_t>("cat4", 0);
  info.MapString<size_t>("cat5", 0);
  info.MapString<size_t>("cat6", 0);
  info.MapString<size_t>("cat0", 1);
  info.MapString<size_t>("cat1", 1);
  info.MapString<size_t>("cat2", 1);
  info.MapString<size_t>("cat0", 2);
  info.MapString<size_t>("cat1", 2);

  // Now generate data.
  arma::Mat<size_t> dataset(3, 9000);
  arma::Row<size_t> labels(9000);
  for (size_t i = 0; i < 9000; i += 3)
  {
    dataset(0, i) = mlpack::math::RandInt(7);
    dataset(1, i) = 0;
    dataset(2, i) = mlpack::math::RandInt(2);
    labels(i) = 0;

    dataset(0, i + 1) = mlpack::math::RandInt(7);
    dataset(1, i + 1) = 2;
    dataset(2, i + 1) = mlpack::math::RandInt(2);
    labels(i + 1) = 1;

    dataset(0, i + 2) = mlpack::math::RandInt(7);
    dataset(1, i + 2) = 1;
    dataset(2, i + 2) = mlpack::math::RandInt(2);
    labels(i + 2) = 2;
  }

  // Now train two streaming decision trees; one on the whole dataset, and one
  // on streaming data.
  typedef HoeffdingTree<GiniImpurity, HoeffdingSizeTNumericSplit,
      HoeffdingCategoricalSplit> TreeType;
  TreeType batchTree(dataset, info, labels, 3, false);
  TreeType streamTree(info, 3);
  for (size_t i = 0; i < 9000; ++i)
    streamTree.Train(dataset.col(i), labels[i]);

  // Each tree should have a single split.
  REQUIRE(batchTree.NumChildren() == 3);
  REQUIRE(streamTree.NumChildren() == 3);
  REQUIRE(batchTree.SplitDimension() == 1);
  REQUIRE(streamTree.SplitDimension() == 1);

  // Now, classify all the points in the dataset.
  arma::Row<size_t> batchLabels(9000);
  arma::Row<size_t> streamLabels(9000);

  streamTree.Classify(dataset, batchLabels);
  for (size_t i = 0; i < 9000; ++i)
    streamLabels[i] = batchTree.Classify(dataset.col(i));

  for (size_t i = 0; i < 9000; ++i)
  {
    REQUIRE(labels[i] == streamLabels[i]);
    REQUIRE(labels[i] == batchLabels[i]);
  }
}

/**
 * Make sure that a tree that does not split on anything.
 */
TEST_CASE("NumDescendantsTest1", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(3, 500);
  arma::Row<size_t> labels(500);
  data::DatasetInfo info(3); // All features are numeric.
  for (size_t i = 0; i <500; i ++)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    labels[i] = 0;
  }

  // Now train streaming decision tree;
  typedef HoeffdingTree<GiniImpurity, HoeffdingDoubleNumericSplit> TreeType;
  TreeType streamTree(info, 3);
  for (size_t i = 0; i < 500; ++i)
    streamTree.Train(dataset.col(i), labels[i]);
  // As there is just one label, there are no descendants.
  REQUIRE(streamTree.NumDescendants() == 0);
}

/**
 * Test that a tree that does split has some descendants.
 */
TEST_CASE("NumDescendantsTest2", "[HoeffdingTreeTest]")
{
  DatasetInfo info(3);
  info.MapString<size_t>("cat0", 0);
  info.MapString<size_t>("cat1", 0);
  info.MapString<size_t>("cat2", 0);
  info.MapString<size_t>("cat3", 0);
  info.MapString<size_t>("cat4", 0);
  info.MapString<size_t>("cat5", 0);
  info.MapString<size_t>("cat6", 0);
  info.MapString<size_t>("cat0", 1);
  info.MapString<size_t>("cat1", 1);
  info.MapString<size_t>("cat2", 1);
  info.MapString<size_t>("cat0", 2);
  info.MapString<size_t>("cat1", 2);
  // Generate data.
  arma::Mat<size_t> dataset(3, 9000);
  arma::Row<size_t> labels(9000);
  for (size_t i = 2; i < 9000; i += 3)
  {
    dataset(0, i) = mlpack::math::RandInt(7);
    dataset(1, i) = 0;
    dataset(2, i) = mlpack::math::RandInt(2);
    labels(i) = 0;

    dataset(0, i - 1) = mlpack::math::RandInt(7);
    dataset(1, i - 1) = 2;
    dataset(2, i - 1) = mlpack::math::RandInt(2);
    labels(i - 1) = 1;

    dataset(0, i - 2) = mlpack::math::RandInt(7);
    dataset(1, i - 2) = 1;
    dataset(2, i - 2) = mlpack::math::RandInt(2);
    labels(i - 2) = 2;
  }

  // Now train the streaming decision tree.  This should split because splitting
  // on dimension 2 gives a perfect split.
  typedef HoeffdingTree<GiniImpurity, HoeffdingSizeTNumericSplit,
      HoeffdingCategoricalSplit> TreeType;
  TreeType batchTree(dataset, info, labels, 3, false);

  REQUIRE(batchTree.NumDescendants() == 3);
}

/**
 * Test that the HoeffdingNumericSplit class has a fitness function value of 0
 * before it's seen enough points.
 */
TEST_CASE("HoeffdingNumericSplitFitnessFunctionTest", "[HoeffdingTreeTest]")
{
  HoeffdingNumericSplit<GiniImpurity> split(5, 10, 100);

  // The first 99 iterations should not calculate anything.  The 100th is where
  // the counting starts.
  for (size_t i = 0; i < 99; ++i)
  {
    split.Train(mlpack::math::Random(), mlpack::math::RandInt(5));
    double bestGain, secondBestGain;
    split.EvaluateFitnessFunction(bestGain, secondBestGain);
    REQUIRE(bestGain == Approx(0.0).margin(1e-10));
    REQUIRE(secondBestGain == Approx(0.0).margin(1e-10));
  }
}

/**
 * Make sure the majority class is correct in the samples before binning.
 */
TEST_CASE("HoeffdingNumericSplitPreBinningMajorityClassTest",
          "[HoeffdingTreeTest]")
{
  HoeffdingNumericSplit<GiniImpurity> split(3, 10, 100);

  for (size_t i = 0; i < 100; ++i)
  {
    split.Train(mlpack::math::Random(), 1);
    REQUIRE(split.MajorityClass() == 1);
  }
}

/**
 * Use a numeric feature that is bimodal (with a margin), and make sure that the
 * HoeffdingNumericSplit bins it reasonably into two bins and returns sensible
 * Gini impurity numbers.
 */
TEST_CASE("HoeffdingNumericSplitBimodalTest", "[HoeffdingTreeTest]")
{
  // 2 classes, 2 bins, 200 samples before binning.
  HoeffdingNumericSplit<GiniImpurity> split(2, 2, 200);

  for (size_t i = 0; i < 100; ++i)
  {
    split.Train(mlpack::math::Random() + 0.3, 0);
    split.Train(-mlpack::math::Random() - 0.3, 1);
  }

  // Push the majority class to 1.
  split.Train(-mlpack::math::Random() - 0.3, 1);
  REQUIRE(split.MajorityClass() == 1);

  // Push the majority class back to 0.
  split.Train(mlpack::math::Random() + 0.3, 0);
  split.Train(mlpack::math::Random() + 0.3, 0);
  REQUIRE(split.MajorityClass() == 0);

  // Now the binning should be complete, and so the impurity should be
  // (0.5 * (1 - 0.5)) * 2 = 0.50 (it will be 0 in the two created children).
  double bestGain, secondBestGain;
  split.EvaluateFitnessFunction(bestGain, secondBestGain);
  REQUIRE(bestGain == Approx(0.50).epsilon(0.0003));
  REQUIRE(secondBestGain == Approx(0.0).margin(1e-10));

  // Make sure that if we do create children, that the correct number of
  // children is created, and that the bins end up in the right place.
  NumericSplitInfo<> info;
  arma::Col<size_t> childMajorities;
  split.Split(childMajorities, info);
  REQUIRE(childMajorities.n_elem == 2);

  // Now check the split info.
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(info.CalculateDirection(mlpack::math::Random() + 0.3) !=
            info.CalculateDirection(-mlpack::math::Random() - 0.3));
  }
}

/**
 * Create a BinaryNumericSplit object, feed it a bunch of samples where anything
 * less than 1.0 is class 0 and anything greater is class 1.  Then make sure it
 * can perform a perfect split.
 */
TEST_CASE("BinaryNumericSplitSimpleSplitTest", "[HoeffdingTreeTest]")
{
  BinaryNumericSplit<GiniImpurity> split(2); // 2 classes.

  // Feed it samples.
  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(mlpack::math::Random(), 0);
    split.Train(mlpack::math::Random() + 1.0, 1);

    // Now ensure the fitness function gives good gain.
    // The Gini impurity for the unsplit node is 2 * (0.5^2) = 0.5, and the Gini
    // impurity for the children is 0.
    double bestGain, secondBestGain;
    split.EvaluateFitnessFunction(bestGain, secondBestGain);
    REQUIRE(bestGain == Approx(0.5).epsilon(1e-7));
    REQUIRE(bestGain > secondBestGain);
  }

  // Now, when we ask it to split, ensure that the split value is reasonable.
  arma::Col<size_t> childMajorities;
  BinaryNumericSplitInfo<> splitInfo;
  split.Split(childMajorities, splitInfo);

  REQUIRE(childMajorities[0] == 0);
  REQUIRE(childMajorities[1] == 1);
  REQUIRE(splitInfo.CalculateDirection(0.5) == 0);
  REQUIRE(splitInfo.CalculateDirection(1.5) == 1);
  REQUIRE(splitInfo.CalculateDirection(0.0) == 0);
  REQUIRE(splitInfo.CalculateDirection(-1.0) == 0);
  REQUIRE(splitInfo.CalculateDirection(0.9) == 0);
  REQUIRE(splitInfo.CalculateDirection(1.1) == 1);
}

/**
 * Create a BinaryNumericSplit object, feed it samples in the same way as
 * before, but with four classes.
 */
TEST_CASE("BinaryNumericSplitSimpleFourClassSplitTest", "[HoeffdingTreeTest]")
{
  BinaryNumericSplit<GiniImpurity> split(4); // 4 classes.

  // Feed it samples.
  for (size_t i = 0; i < 250; ++i)
  {
    split.Train(mlpack::math::Random(), 0);
    split.Train(mlpack::math::Random() + 2.0, 1);
    split.Train(mlpack::math::Random() - 1.0, 2);
    split.Train(mlpack::math::Random() + 1.0, 3);

    // The same as the previous test, but with four classes: 4 * (0.25 * 0.75) =
    // 0.75.  We can only split in one place, though, which will give one
    // perfect child, giving a gain of 0.75 - 3 * (1/3 * 2/3) = 0.25.
    double bestGain, secondBestGain;
    split.EvaluateFitnessFunction(bestGain, secondBestGain);
    REQUIRE(bestGain == Approx(0.25).epsilon(1e-7));
    REQUIRE(bestGain >= secondBestGain);
  }

  // Now, when we ask it to split, ensure that the split value is reasonable.
  arma::Col<size_t> childMajorities;
  BinaryNumericSplitInfo<> splitInfo;
  split.Split(childMajorities, splitInfo);

  // We don't really care where it splits -- it can split anywhere.  But it has
  // to split in only two directions.
  REQUIRE(childMajorities.n_elem == 2);
}

/**
 * Create a HoeffdingTree that uses the HoeffdingNumericSplit and make sure it
 * can split meaningfully on the correct dimension.
 */
TEST_CASE("NumericHoeffdingTreeTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(3, 9000);
  arma::Row<size_t> labels(9000);
  data::DatasetInfo info(3); // All features are numeric.
  for (size_t i = 0; i < 9000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    labels[i + 2] = 1;
  }

  // Now train two streaming decision trees; one on the whole dataset, and one
  // on streaming data.
  typedef HoeffdingTree<GiniImpurity, HoeffdingDoubleNumericSplit> TreeType;
  TreeType batchTree(dataset, info, labels, 3, false);
  TreeType streamTree(info, 3);
  for (size_t i = 0; i < 9000; ++i)
    streamTree.Train(dataset.col(i), labels[i]);

  // Each tree should have at least one split.
  REQUIRE(batchTree.NumChildren() > 0);
  REQUIRE(streamTree.NumChildren() > 0);
  REQUIRE(batchTree.SplitDimension() == 1);
  REQUIRE(streamTree.SplitDimension() == 1);

  // Now, classify all the points in the dataset.
  arma::Row<size_t> batchLabels(9000);
  arma::Row<size_t> streamLabels(9000);

  streamTree.Classify(dataset, batchLabels);
  for (size_t i = 0; i < 9000; ++i)
    streamLabels[i] = batchTree.Classify(dataset.col(i));

  size_t streamCorrect = 0;
  size_t batchCorrect = 0;
  for (size_t i = 0; i < 9000; ++i)
  {
    if (labels[i] == streamLabels[i])
      ++streamCorrect;
    if (labels[i] == batchLabels[i])
      ++batchCorrect;
  }

  // 66% accuracy shouldn't be too much to ask...
  REQUIRE(streamCorrect > 6000);
  REQUIRE(batchCorrect > 6000);
}

/**
 * The same as the previous test, but with the numeric binary split, and with a
 * categorical feature.
 */
TEST_CASE("BinaryNumericHoeffdingTreeTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(4, 9000);
  arma::Row<size_t> labels(9000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString<double>("0", 3);
  for (size_t i = 0; i < 9000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    dataset(3, i) = 0.0;
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    dataset(3, i + 1) = 0.0;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    dataset(3, i + 2) = 0.0;
    labels[i + 2] = 1;
  }

  // Now train two streaming decision trees; one on the whole dataset, and one
  // on streaming data.
  typedef HoeffdingTree<GiniImpurity, BinaryDoubleNumericSplit> TreeType;
  TreeType batchTree(dataset, info, labels, 3, false);
  TreeType streamTree(info, 3);
  for (size_t i = 0; i < 9000; ++i)
    streamTree.Train(dataset.col(i), labels[i]);

  // Each tree should have at least one split.
  REQUIRE(batchTree.NumChildren() > 0);
  REQUIRE(streamTree.NumChildren() > 0);
  REQUIRE(batchTree.SplitDimension() == 1);
  REQUIRE(streamTree.SplitDimension() == 1);

  // Now, classify all the points in the dataset.
  arma::Row<size_t> batchLabels(9000);
  arma::Row<size_t> streamLabels(9000);

  streamTree.Classify(dataset, batchLabels);
  for (size_t i = 0; i < 9000; ++i)
    streamLabels[i] = batchTree.Classify(dataset.col(i));

  size_t streamCorrect = 0;
  size_t batchCorrect = 0;
  for (size_t i = 0; i < 9000; ++i)
  {
    if (labels[i] == streamLabels[i])
      ++streamCorrect;
    if (labels[i] == batchLabels[i])
      ++batchCorrect;
  }

  // Require a pretty high accuracy: 95%.
  REQUIRE(streamCorrect > 8550);
  REQUIRE(batchCorrect > 8550);
}

/**
 * Test majority probabilities.
 */
TEST_CASE("MajorityProbabilityTest", "[HoeffdingTreeTest]")
{
  data::DatasetInfo info(1);
  HoeffdingTree<> tree(info, 3);

  // Feed the tree a few samples.
  tree.Train(arma::vec("1"), 0);
  tree.Train(arma::vec("2"), 0);
  tree.Train(arma::vec("3"), 0);

  size_t prediction;
  double probability;
  tree.Classify(arma::vec("1"), prediction, probability);

  REQUIRE(prediction == 0);
  REQUIRE(probability == Approx(1.0).epsilon(1e-7));

  // Make it impure.
  tree.Train(arma::vec("4"), 1);
  tree.Classify(arma::vec("3"), prediction, probability);

  REQUIRE(prediction == 0);
  REQUIRE(probability == Approx(0.75).epsilon(1e-7));

  // Flip the majority class.
  tree.Train(arma::vec("4"), 1);
  tree.Train(arma::vec("4"), 1);
  tree.Train(arma::vec("4"), 1);
  tree.Train(arma::vec("4"), 1);
  tree.Classify(arma::vec("3"), prediction, probability);

  REQUIRE(prediction == 1);
  REQUIRE(probability == Approx(0.625).epsilon(1e-7));
}

/**
 * Make sure that batch training mode outperforms non-batch mode.
 */
TEST_CASE("BatchTrainingTest", "[HoeffdingTreeTest]")
{
  // We need to create a dataset with some amount of complexity, that must be
  // split in a handful of ways to accurately classify the data.  An expanding
  // spiral should do the trick here.  We'll make the spiral in two dimensions.
  // The label will change as the index increases.
  arma::mat spiralDataset(2, 10000);
  for (size_t i = 0; i < 10000; ++i)
  {
    // One circle every 20000 samples.  Plus some noise.
    const double magnitude = 2.0 + (double(i) / 20000.0) +
        0.5 * mlpack::math::Random();
    const double angle = (i % 20000) * (2 * M_PI) + mlpack::math::Random();

    const double x = magnitude * cos(angle);
    const double y = magnitude * sin(angle);

    spiralDataset(0, i) = x;
    spiralDataset(1, i) = y;
  }

  arma::Row<size_t> labels(10000);
  for (size_t i = 0; i < 2000; ++i)
    labels[i] = 1;
  for (size_t i = 2000; i < 4000; ++i)
    labels[i] = 3;
  for (size_t i = 4000; i < 6000; ++i)
    labels[i] = 2;
  for (size_t i = 6000; i < 8000; ++i)
    labels[i] = 0;
  for (size_t i = 8000; i < 10000; ++i)
    labels[i] = 4;

  // Now shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, 9999,
      10000));
  arma::mat d(2, 10000);
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

  data::DatasetInfo info(2);

  // Now build two decision trees; one in batch mode, and one in streaming mode.
  // We need to set the confidence pretty high so that the streaming tree isn't
  // able to have enough samples to build to the same leaves.
  HoeffdingTree<> batchTree(trainingData, info, trainingLabels, 5, true,
      0.99999999);
  HoeffdingTree<> streamTree(trainingLabels, info, trainingLabels, 5, false,
      0.99999999);

  // Ensure that the performance of the batch tree is better.
  size_t batchCorrect = 0;
  size_t streamCorrect = 0;
  for (size_t i = 0; i < 5000; ++i)
  {
    size_t streamLabel = streamTree.Classify(testData.col(i));
    size_t batchLabel = batchTree.Classify(testData.col(i));

    if (streamLabel == testLabels[i])
      ++streamCorrect;
    if (batchLabel == testLabels[i])
      ++batchCorrect;
  }

  // The batch tree must be a bit better than the stream tree.  But not too
  // much, since the accuracy is already going to be very high.
  REQUIRE(batchCorrect >= streamCorrect);
}

// Make sure that changing the confidence properly propagates to all leaves.
TEST_CASE("ConfidenceChangeTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(4, 9000);
  arma::Row<size_t> labels(9000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString<double>("0", 3);
  for (size_t i = 0; i < 9000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    dataset(3, i) = 0.0;
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    dataset(3, i + 1) = 0.0;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    dataset(3, i + 2) = 0.0;
    labels[i + 2] = 1;
  }

  HoeffdingTree<> tree(info, 3, 0.5); // Low success probability.

  size_t i = 0;
  while ((tree.NumChildren() == 0) && (i < 9000))
  {
    tree.Train(dataset.col(i), labels[i]);
    ++i;
  }

  REQUIRE(i < 9000);

  // Now we have split the root node, but we need to make sure we can feed
  // through the rest of the points while requiring a confidence of 1.0, and
  // make sure no splits happen.
  tree.SuccessProbability(1.0);
  tree.MaxSamples(0);

  i = 0;
  while ((tree.NumChildren() == 0) && (i < 90000))
  {
    tree.Train(dataset.col(i % 9000), labels[i % 9000]);
    ++i;
  }

  for (size_t c = 0; c < tree.NumChildren(); ++c)
    REQUIRE(tree.Child(c).NumChildren() == 0);
}

//! Make sure parameter changes are propagated to children.
TEST_CASE("ParameterChangeTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(4, 9000);
  arma::Row<size_t> labels(9000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString<double>("0", 3);
  for (size_t i = 0; i < 9000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    dataset(3, i) = 0.0;
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    dataset(3, i + 1) = 0.0;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    dataset(3, i + 2) = 0.0;
    labels[i + 2] = 1;
  }

  HoeffdingTree<> tree(dataset, info, labels, 3, true); // Batch training.

  // Now change parameters...
  tree.SuccessProbability(0.7);
  tree.MinSamples(17);
  tree.MaxSamples(192);
  tree.CheckInterval(3);

  std::stack<HoeffdingTree<>*> stack;
  stack.push(&tree);
  while (!stack.empty())
  {
    HoeffdingTree<>* node = stack.top();
    stack.pop();

    REQUIRE(node->SuccessProbability() == Approx(0.7).epsilon(1e-7));
    REQUIRE(node->MinSamples() == 17);
    REQUIRE(node->MaxSamples() == 192);
    REQUIRE(node->CheckInterval() == 3);

    for (size_t i = 0; i < node->NumChildren(); ++i)
      stack.push(&node->Child(i));
  }
}

TEST_CASE("MultipleSerializationTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(4, 9000);
  arma::Row<size_t> labels(9000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString<double>("0", 3);
  for (size_t i = 0; i < 9000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    dataset(3, i) = 0.0;
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    dataset(3, i + 1) = 0.0;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    dataset(3, i + 2) = 0.0;
    labels[i + 2] = 1;
  }

  // Batch training will give a tree with many labels.
  HoeffdingTree<> deepTree(dataset, info, labels, 3, true);
  // Streaming training will not.
  HoeffdingTree<> shallowTree(dataset, info, labels, 3, false);

  // Now serialize the shallow tree into the deep tree.
  std::ostringstream oss;
  {
    boost::archive::binary_oarchive boa(oss);
    boa << BOOST_SERIALIZATION_NVP(shallowTree);
  }

  std::istringstream iss(oss.str());
  {
    boost::archive::binary_iarchive bia(iss);
    bia >> BOOST_SERIALIZATION_NVP(deepTree);
  }

  // Now do some classification and make sure the results are the same.
  arma::Row<size_t> deepPredictions, shallowPredictions;
  shallowTree.Classify(dataset, shallowPredictions);
  deepTree.Classify(dataset, deepPredictions);

  for (size_t i = 0; i < deepPredictions.n_elem; ++i)
  {
    REQUIRE(shallowPredictions[i] == deepPredictions[i]);
  }
}

// Test the Hoeffding tree model.
TEST_CASE("HoeffdingTreeModelTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(4, 3000);
  arma::Row<size_t> labels(3000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString<double>("0", 3);
  for (size_t i = 0; i < 3000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    dataset(3, i) = 0.0;
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    dataset(3, i + 1) = 0.0;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    dataset(3, i + 2) = 0.0;
    labels[i + 2] = 1;
  }

  // Train a model on a simple dataset, for all four types of models, and make
  // sure we get reasonable results.
  for (size_t i = 0; i < 4; ++i)
  {
    HoeffdingTreeModel m;
    switch (i)
    {
      case 0:
        m = HoeffdingTreeModel(HoeffdingTreeModel::GINI_HOEFFDING);
        break;

      case 1:
        m = HoeffdingTreeModel(HoeffdingTreeModel::GINI_BINARY);
        break;

      case 2:
        m = HoeffdingTreeModel(HoeffdingTreeModel::INFO_HOEFFDING);
        break;

      case 3:
        m = HoeffdingTreeModel(HoeffdingTreeModel::INFO_BINARY);
        break;
    }

    // We'll take 5 passes over the data.
    m.BuildModel(dataset, info, labels, 3, false, 0.99, 1000, 100, 100, 4, 100);
    for (size_t j = 0; j < 4; ++j)
      m.Train(dataset, labels, false);

    // Now make sure the performance is reasonable.
    arma::Row<size_t> predictions, predictions2;
    arma::rowvec probabilities;
    m.Classify(dataset, predictions);
    m.Classify(dataset, predictions2, probabilities);

    size_t correct = 0;
    for (size_t i = 0; i < 3000; ++i)
    {
      // Check consistency of predictions.
      REQUIRE(predictions[i] == predictions2[i]);

      if (labels[i] == predictions[i])
        ++correct;
    }

    // Require at least 95% accuracy.
    REQUIRE(correct > 2850);
  }
}

// Test the Hoeffding tree model in batch mode.
TEST_CASE("HoeffdingTreeModelBatchTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(4, 3000);
  arma::Row<size_t> labels(3000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString<double>("0", 3);
  for (size_t i = 0; i < 3000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    dataset(3, i) = 0.0;
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    dataset(3, i + 1) = 0.0;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    dataset(3, i + 2) = 0.0;
    labels[i + 2] = 1;
  }

  // Train a model on a simple dataset, for all four types of models, and make
  // sure we get reasonable results.
  for (size_t i = 0; i < 4; ++i)
  {
    HoeffdingTreeModel m;
    switch (i)
    {
      case 0:
        m = HoeffdingTreeModel(HoeffdingTreeModel::GINI_HOEFFDING);
        break;

      case 1:
        m = HoeffdingTreeModel(HoeffdingTreeModel::GINI_BINARY);
        break;

      case 2:
        m = HoeffdingTreeModel(HoeffdingTreeModel::INFO_HOEFFDING);
        break;

      case 3:
        m = HoeffdingTreeModel(HoeffdingTreeModel::INFO_BINARY);
        break;
    }

    // Train in batch.
    m.BuildModel(dataset, info, labels, 3, true, 0.99, 1000, 100, 100, 4, 100);

    // Now make sure the performance is reasonable.
    arma::Row<size_t> predictions, predictions2;
    arma::rowvec probabilities;
    m.Classify(dataset, predictions);
    m.Classify(dataset, predictions2, probabilities);

    size_t correct = 0;
    for (size_t i = 0; i < 3000; ++i)
    {
      // Check consistency of predictions.
      REQUIRE(predictions[i] == predictions2[i]);

      if (labels[i] == predictions[i])
        ++correct;
    }

    // Require at least 95% accuracy.
    REQUIRE(correct > 2850);
  }
}

TEST_CASE("HoeffdingTreeModelSerializationTest", "[HoeffdingTreeTest]")
{
  // Generate data.
  arma::mat dataset(4, 3000);
  arma::Row<size_t> labels(3000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString<double>("0", 3);
  for (size_t i = 0; i < 3000; i += 3)
  {
    dataset(0, i) = mlpack::math::Random();
    dataset(1, i) = mlpack::math::Random();
    dataset(2, i) = mlpack::math::Random();
    dataset(3, i) = 0.0;
    labels[i] = 0;

    dataset(0, i + 1) = mlpack::math::Random();
    dataset(1, i + 1) = mlpack::math::Random() - 1.0;
    dataset(2, i + 1) = mlpack::math::Random() + 0.5;
    dataset(3, i + 1) = 0.0;
    labels[i + 1] = 2;

    dataset(0, i + 2) = mlpack::math::Random();
    dataset(1, i + 2) = mlpack::math::Random() + 1.0;
    dataset(2, i + 2) = mlpack::math::Random() + 0.8;
    dataset(3, i + 2) = 0.0;
    labels[i + 2] = 1;
  }

  // Train a model on a simple dataset, for all four types of models, and make
  // sure we get reasonable results.
  for (size_t i = 0; i < 4; ++i)
  {
    HoeffdingTreeModel m, xmlM, textM, binaryM;
    switch (i)
    {
      case 0:
        m = HoeffdingTreeModel(HoeffdingTreeModel::GINI_HOEFFDING);
        break;

      case 1:
        m = HoeffdingTreeModel(HoeffdingTreeModel::GINI_BINARY);
        break;

      case 2:
        m = HoeffdingTreeModel(HoeffdingTreeModel::INFO_HOEFFDING);
        break;

      case 3:
        m = HoeffdingTreeModel(HoeffdingTreeModel::INFO_BINARY);
        break;
    }

    // Train in batch.
    m.BuildModel(dataset, info, labels, 3, true, 0.99, 1000, 100, 100, 4, 100);
    // False training of XML model.
    xmlM.BuildModel(dataset, info, labels, 3, false, 0.5, 100, 100, 100, 2,
        100);

    // Now make sure the performance is reasonable.
    arma::Row<size_t> predictions, predictionsXml, predictionsText,
        predictionsBinary;
    arma::rowvec probabilities, probabilitiesXml, probabilitiesText,
        probabilitiesBinary;

    SerializeObjectAll(m, xmlM, textM, binaryM);

    // Get predictions for all.
    m.Classify(dataset, predictions, probabilities);
    xmlM.Classify(dataset, predictionsXml, probabilitiesXml);
    textM.Classify(dataset, predictionsText, probabilitiesText);
    binaryM.Classify(dataset, predictionsBinary, probabilitiesBinary);

    for (size_t i = 0; i < 3000; ++i)
    {
      // Check consistency of predictions and probabilities.
      REQUIRE(predictions[i] == predictionsXml[i]);
      REQUIRE(predictions[i] == predictionsText[i]);
      REQUIRE(predictions[i] == predictionsBinary[i]);

      REQUIRE(probabilities[i] == Approx(probabilitiesXml[i]).epsilon(1e-7));
      REQUIRE(probabilities[i] == Approx(probabilitiesText[i]).epsilon(1e-7));
      REQUIRE(probabilities[i] == Approx(probabilitiesBinary[i]).epsilon(1e-7));
    }
  }
}
