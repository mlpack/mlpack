/**
 * @file hoeffding_tree_test.cpp
 * @author Ryan Curtin
 *
 * Test file for Hoeffding trees.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/streaming_decision_tree.hpp>
#include <mlpack/methods/hoeffding_trees/gini_impurity.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_split.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_categorical_split.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::data;
using namespace mlpack::tree;

BOOST_AUTO_TEST_SUITE(HoeffdingTreeTest);

BOOST_AUTO_TEST_CASE(GiniImpurityPerfectSimpleTest)
{
  // Make a simple test for Gini impurity with one class.  In this case it
  // should always be 0.  We'll assemble the count matrix by hand.
  arma::Mat<size_t> counts(2, 2); // 2 categories, 2 classes.

  counts(0, 0) = 10; // 10 points in category 0 with class 0.
  counts(0, 1) = 0; // 0 points in category 0 with class 1.
  counts(1, 0) = 12; // 12 points in category 1 with class 0.
  counts(1, 1) = 0; // 0 points in category 1 with class 1.

  // Since the split gets us nothing, there should be no gain.
  BOOST_REQUIRE_SMALL(GiniImpurity::Evaluate(counts), 1e-10);
}

BOOST_AUTO_TEST_CASE(GiniImpurityImperfectSimpleTest)
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
  BOOST_REQUIRE_CLOSE(GiniImpurity::Evaluate(counts), 0.5, 1e-5);
}

BOOST_AUTO_TEST_CASE(GiniImpurityBadSplitTest)
{
  // Make a simple test where a split gets us nothing.
  arma::Mat<size_t> counts(2, 2);
  counts(0, 0) = 10;
  counts(0, 1) = 10;
  counts(1, 0) = 5;
  counts(1, 1) = 5;

  BOOST_REQUIRE_SMALL(GiniImpurity::Evaluate(counts), 1e-10);
}

/**
 * A hand-crafted more difficult test for the Gini impurity, where four
 * categories and three classes are available.
 */
BOOST_AUTO_TEST_CASE(GiniImpurityThreeClassTest)
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
  BOOST_REQUIRE_CLOSE(GiniImpurity::Evaluate(counts), 0.26145, 1e-3);
}

BOOST_AUTO_TEST_CASE(GiniImpurityZeroTest)
{
  // When nothing has been seen, the gini impurity should be zero.
  arma::Mat<size_t> counts = arma::zeros<arma::Mat<size_t>>(10, 10);

  BOOST_REQUIRE_SMALL(GiniImpurity::Evaluate(counts), 1e-10);
}

/**
 * Test that the range of Gini impurities is correct for a handful of class
 * sizes.
 */
BOOST_AUTO_TEST_CASE(GiniImpurityRangeTest)
{
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(1), 0, 1e-5);
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(2), 0.5, 1e-5);
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(3), 0.66666667, 1e-5);
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(4), 0.75, 1e-5);
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(5), 0.8, 1e-5);
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(10), 0.9, 1e-5);
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(100), 0.99, 1e-5);
  BOOST_REQUIRE_CLOSE(GiniImpurity::Range(1000), 0.999, 1e-5);
}

/**
 * Feed the HoeffdingCategoricalSplit class many examples, all from the same
 * class, and verify that the majority class is correct.
 */
BOOST_AUTO_TEST_CASE(HoeffdingCategoricalSplitMajorityClassTest)
{
  // Ten categories, three classes.
  HoeffdingCategoricalSplit<GiniImpurity> split(10, 3);

  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(mlpack::math::RandInt(0, 10), 1);
    BOOST_REQUIRE_EQUAL(split.MajorityClass(), 1);
  }
}

/**
 * A harder majority class example.
 */
BOOST_AUTO_TEST_CASE(HoeffdingCategoricalSplitHarderMajorityClassTest)
{
  // Ten categories, three classes.
  HoeffdingCategoricalSplit<GiniImpurity> split(10, 3);

  split.Train(mlpack::math::RandInt(0, 10), 1);
  for (size_t i = 0; i < 250; ++i)
  {
    split.Train(mlpack::math::RandInt(0, 10), 1);
    split.Train(mlpack::math::RandInt(0, 10), 2);
    BOOST_REQUIRE_EQUAL(split.MajorityClass(), 1);
  }
}

/**
 * Ensure that the fitness function is positive when we pass some data that
 * would result in an improvement if it was split.
 */
BOOST_AUTO_TEST_CASE(HoeffdingCategoricalSplitEasyFitnessCheck)
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

  BOOST_REQUIRE_GT(split.EvaluateFitnessFunction(), 0.0);
}

/**
 * Ensure that the fitness function returns 0 (no improvement) when a split
 * would not get us any improvement.
 */
BOOST_AUTO_TEST_CASE(HoeffdingCategoricalSplitNoImprovementFitnessTest)
{
  HoeffdingCategoricalSplit<GiniImpurity> split(2, 2);

  // No training has yet happened, so a split would get us nothing.
  BOOST_REQUIRE_SMALL(split.EvaluateFitnessFunction(), 1e-10);

  split.Train(0, 0);
  split.Train(1, 0);
  split.Train(0, 1);
  split.Train(1, 1);

  // Now, a split still gets us only 50% accuracy in each split bin.
  BOOST_REQUIRE_SMALL(split.EvaluateFitnessFunction(), 1e-10);
}

/**
 * Test that when we do split, we get reasonable split information.
 */
BOOST_AUTO_TEST_CASE(HoeffdingCategoricalSplitSplitTest)
{
  HoeffdingCategoricalSplit<GiniImpurity> split(3, 3); // 3 categories.

  // No training is necessary because we can just call CreateChildren().
  std::vector<StreamingDecisionTree<HoeffdingSplit<>>> children;
  data::DatasetInfo info(3);
  info.MapString("hello", 0); // Make dimension 0 categorical.
  HoeffdingCategoricalSplit<GiniImpurity>::SplitInfo splitInfo(3);

  // Create the children.
  arma::Col<size_t> childMajorities;
  split.Split(childMajorities, splitInfo);

  BOOST_REQUIRE_EQUAL(childMajorities.n_elem, 3);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(0), 0);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(1), 1);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(2), 2);
}

/**
 * If we feed the HoeffdingSplit a ton of points of the same class, it should
 * not suggest that we split.
 */
BOOST_AUTO_TEST_CASE(HoeffdingSplitNoSplitTest)
{
  // Make all dimensions categorical.
  data::DatasetInfo info(3);
  info.MapString("cat1", 0);
  info.MapString("cat2", 0);
  info.MapString("cat3", 0);
  info.MapString("cat4", 0);
  info.MapString("cat1", 1);
  info.MapString("cat2", 1);
  info.MapString("cat3", 1);
  info.MapString("cat1", 2);
  info.MapString("cat2", 2);

  HoeffdingSplit<> split(3, 2, info, 0.95, 5000, 1);

  // Feed it samples.
  for (size_t i = 0; i < 1000; ++i)
  {
    // Create the test point.
    arma::Col<size_t> testPoint(3);
    testPoint(0) = mlpack::math::RandInt(0, 4);
    testPoint(1) = mlpack::math::RandInt(0, 3);
    testPoint(2) = mlpack::math::RandInt(0, 2);
    split.Train(testPoint, 0); // Always label 0.

    BOOST_REQUIRE_EQUAL(split.SplitCheck(), 0);
  }
}

/**
 * If we feed the HoeffdingSplit a ton of points of two different classes, it
 * should very clearly suggest that we split (eventually).
 */
BOOST_AUTO_TEST_CASE(HoeffdingSplitEasySplitTest)
{
  // It'll be a two-dimensional dataset with two categories each.  In the first
  // dimension, category 0 will only receive points with class 0, and category 1
  // will only receive points with class 1.  In the second dimension, all points
  // will have category 0 (so it is useless).
  data::DatasetInfo info(2);
  info.MapString("cat0", 0);
  info.MapString("cat1", 0);
  info.MapString("cat0", 1);

  HoeffdingSplit<> split(2, 2, info, 0.95, 5000, 1);

  // Feed samples from each class.
  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(arma::Col<size_t>("0 0"), 0);
    split.Train(arma::Col<size_t>("1 0"), 1);
  }

  // Now it should be ready to split.
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), 2);
  BOOST_REQUIRE_EQUAL(split.SplitDimension(), 0);
}

/**
 * If we force a success probability of 1, it should never split.
 */
BOOST_AUTO_TEST_CASE(HoeffdingSplitProbability1SplitTest)
{
  // It'll be a two-dimensional dataset with two categories each.  In the first
  // dimension, category 0 will only receive points with class 0, and category 1
  // will only receive points with class 1.  In the second dimension, all points
  // will have category 0 (so it is useless).
  data::DatasetInfo info(2);
  info.MapString("cat0", 0);
  info.MapString("cat1", 0);
  info.MapString("cat0", 1);

  HoeffdingSplit<> split(2, 2, info, 1.0, 12000, 1);

  // Feed samples from each class.
  for (size_t i = 0; i < 5000; ++i)
  {
    split.Train(arma::Col<size_t>("0 0"), 0);
    split.Train(arma::Col<size_t>("1 0"), 1);
  }

  // But because the success probability is 1, it should never split.
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), 0);
  BOOST_REQUIRE_EQUAL(split.SplitDimension(), size_t(-1));
}

/**
 * A slightly harder splitting problem: there are two features; one gives
 * perfect classification, another gives almost perfect classification (with 10%
 * error).  Splits should occur after many samples.
 */
BOOST_AUTO_TEST_CASE(HoeffdingSplitAlmostPerfectSplit)
{
  // Two categories and two dimensions.
  data::DatasetInfo info(2);
  info.MapString("cat0", 0);
  info.MapString("cat1", 0);
  info.MapString("cat0", 1);
  info.MapString("cat1", 1);

  HoeffdingSplit<> split(2, 2, info, 0.95, 5000, 1);

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
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), 2);
  // Make sure that it's split on the correct dimension.
  BOOST_REQUIRE_EQUAL(split.SplitDimension(), 1);
}

/**
 * Test that the HoeffdingSplit class will not split if the two features are
 * equally good.
 */
BOOST_AUTO_TEST_CASE(HoeffdingSplitEqualSplitTest)
{
  // Two categories and two dimensions.
  data::DatasetInfo info(2);
  info.MapString("cat0", 0);
  info.MapString("cat1", 0);
  info.MapString("cat0", 1);
  info.MapString("cat1", 1);

  HoeffdingSplit<> split(2, 2, info, 0.95, 5000, 1);

  // Feed samples.
  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(arma::Col<size_t>("0 0"), 0);
    split.Train(arma::Col<size_t>("1 1"), 1);
  }

  // Ensure that splitting should not happen.
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), 0);
}

/**
 * Build a decision tree on a dataset with two meaningless dimensions and ensure
 * that it can properly classify all of the training points.  (The dataset is
 * perfectly separable.)
 */
BOOST_AUTO_TEST_CASE(StreamingDecisionTreeSimpleDatasetTest)
{
  DatasetInfo info(3);
  info.MapString("cat0", 0);
  info.MapString("cat1", 0);
  info.MapString("cat2", 0);
  info.MapString("cat3", 0);
  info.MapString("cat4", 0);
  info.MapString("cat5", 0);
  info.MapString("cat6", 0);
  info.MapString("cat0", 1);
  info.MapString("cat1", 1);
  info.MapString("cat2", 1);
  info.MapString("cat0", 2);
  info.MapString("cat1", 2);

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
  StreamingDecisionTree<HoeffdingSplit<GiniImpurity,
      HoeffdingDoubleNumericSplit>, arma::Mat<size_t>>
      batchTree(dataset, info, labels, 3);
  StreamingDecisionTree<HoeffdingSplit<GiniImpurity,
      HoeffdingDoubleNumericSplit>, arma::Mat<size_t>>
      streamTree(info, 3, 3);
  for (size_t i = 0; i < 9000; ++i)
    streamTree.Train(dataset.col(i), labels[i]);

  // Each tree should have a single split.
  BOOST_REQUIRE_EQUAL(batchTree.NumChildren(), 3);
  BOOST_REQUIRE_EQUAL(streamTree.NumChildren(), 3);
  BOOST_REQUIRE_EQUAL(batchTree.Split().SplitDimension(), 1);
  BOOST_REQUIRE_EQUAL(streamTree.Split().SplitDimension(), 1);

  // Now, classify all the points in the dataset.
  arma::Row<size_t> batchLabels(9000);
  arma::Row<size_t> streamLabels(9000);

  streamTree.Classify(dataset, batchLabels);
  for (size_t i = 0; i < 9000; ++i)
    streamLabels[i] = batchTree.Classify(dataset.col(i));

  for (size_t i = 0; i < 9000; ++i)
  {
    BOOST_REQUIRE_EQUAL(labels[i], streamLabels[i]);
    BOOST_REQUIRE_EQUAL(labels[i], batchLabels[i]);
  }
}

/**
 * Test that the HoeffdingNumericSplit class has a fitness function value of 0
 * before it's seen enough points.
 */
BOOST_AUTO_TEST_CASE(HoeffdingNumericSplitFitnessFunctionTest)
{
  HoeffdingNumericSplit<GiniImpurity> split(5, 10, 100);

  // The first 99 iterations should not calculate anything.  The 100th is where
  // the counting starts.
  for (size_t i = 0; i < 99; ++i)
  {
    split.Train(mlpack::math::Random(), mlpack::math::RandInt(5));
    BOOST_REQUIRE_SMALL(split.EvaluateFitnessFunction(), 1e-10);
  }
}

/**
 * Make sure the majority class is correct in the samples before binning.
 */
BOOST_AUTO_TEST_CASE(HoeffdingNumericSplitPreBinningMajorityClassTest)
{
  HoeffdingNumericSplit<GiniImpurity> split(3, 10, 100);

  for (size_t i = 0; i < 100; ++i)
  {
    split.Train(mlpack::math::Random(), 1);
    BOOST_REQUIRE_EQUAL(split.MajorityClass(), 1);
  }
}

/**
 * Use a numeric feature that is bimodal (with a margin), and make sure that the
 * HoeffdingNumericSplit bins it reasonably into two bins and returns sensible
 * Gini impurity numbers.
 */
BOOST_AUTO_TEST_CASE(HoeffdingNumericSplitBimodalTest)
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
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), 1);

  // Push the majority class back to 0.
  split.Train(mlpack::math::Random() + 0.3, 0);
  split.Train(mlpack::math::Random() + 0.3, 0);
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), 0);

  // Now the binning should be complete, and so the impurity should be
  // (0.5 * (1 - 0.5)) * 2 = 0.50 (it will be 0 in the two created children).
  BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(), 0.50, 0.03);

  // Make sure that if we do create children, that the correct number of
  // children is created, and that the bins end up in the right place.
  NumericSplitInfo<> info;
  arma::Col<size_t> childMajorities;
  split.Split(childMajorities, info);
  BOOST_REQUIRE_EQUAL(childMajorities.n_elem, 2);

  // Now check the split info.
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_NE(info.CalculateDirection(mlpack::math::Random() + 0.3),
                     info.CalculateDirection(-mlpack::math::Random() - 0.3));
  }
}

/**
 * Create a BinaryNumericSplit object, feed it a bunch of samples where anything
 * less than 1.0 is class 0 and anything greater is class 1.  Then make sure it
 * can perform a perfect split.
 */
BOOST_AUTO_TEST_CASE(BinaryNumericSplitSimpleSplitTest)
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
    BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(), 0.5, 1e-5);
  }

  // Now, when we ask it to split, ensure that the split value is reasonable.
  arma::Col<size_t> childMajorities;
  NumericSplitInfo<> splitInfo;
  split.Split(childMajorities, splitInfo);

  BOOST_REQUIRE_EQUAL(childMajorities[0], 0);
  BOOST_REQUIRE_EQUAL(childMajorities[1], 1);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(0.5), 0);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(1.5), 1);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(0.0), 0);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(-1.0), 0);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(0.9), 0);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(1.1), 1);
}

/**
 * Create a BinaryNumericSplit object, feed it samples in the same way as
 * before, but with four classes.
 */
BOOST_AUTO_TEST_CASE(BinaryNumericSplitSimpleFourClassSplitTest)
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
    BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(), 0.25, 1e-5);
  }

  // Now, when we ask it to split, ensure that the split value is reasonable.
  arma::Col<size_t> childMajorities;
  NumericSplitInfo<> splitInfo;
  split.Split(childMajorities, splitInfo);

  // We don't really care where it splits -- it can split anywhere.  But it has
  // to split in only two directions.
  BOOST_REQUIRE_EQUAL(childMajorities.n_elem, 2);
}

/**
 * Create a StreamingDecisionTree that uses the HoeffdingNumericSplit and make
 * sure it can split meaningfully on the correct dimension.
 */
BOOST_AUTO_TEST_CASE(NumericHoeffdingTreeTest)
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
  StreamingDecisionTree<HoeffdingSplit<GiniImpurity,
      HoeffdingDoubleNumericSplit>, arma::mat> batchTree(dataset, info, labels,
      3);
  StreamingDecisionTree<HoeffdingSplit<GiniImpurity,
      HoeffdingDoubleNumericSplit>, arma::mat> streamTree(info, 3, 3);
  for (size_t i = 0; i < 9000; ++i)
    streamTree.Train(dataset.col(i), labels[i]);

  // Each tree should have at least one split.
  BOOST_REQUIRE_GT(batchTree.NumChildren(), 0);
  BOOST_REQUIRE_GT(streamTree.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(batchTree.Split().SplitDimension(), 1);
  BOOST_REQUIRE_EQUAL(streamTree.Split().SplitDimension(), 1);

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
  BOOST_REQUIRE_GT(streamCorrect, 6000);
  BOOST_REQUIRE_GT(batchCorrect, 6000);
}

/**
 * The same as the previous test, but with the numeric binary split, and with a
 * categorical feature.
 */
BOOST_AUTO_TEST_CASE(BinaryNumericHoeffdingTreeTest)
{
  // Generate data.
  arma::mat dataset(4, 9000);
  arma::Row<size_t> labels(9000);
  data::DatasetInfo info(4); // All features are numeric, except the fourth.
  info.MapString("0", 3);
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
  StreamingDecisionTree<HoeffdingSplit<GiniImpurity, BinaryDoubleNumericSplit>,
      arma::mat> batchTree(dataset, info, labels, 3);
  StreamingDecisionTree<HoeffdingSplit<GiniImpurity, BinaryDoubleNumericSplit>,
      arma::mat> streamTree(info, 4, 3);
  for (size_t i = 0; i < 9000; ++i)
    streamTree.Train(dataset.col(i), labels[i]);

  // Each tree should have at least one split.
  BOOST_REQUIRE_GT(batchTree.NumChildren(), 0);
  BOOST_REQUIRE_GT(streamTree.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(batchTree.Split().SplitDimension(), 1);
  BOOST_REQUIRE_EQUAL(streamTree.Split().SplitDimension(), 1);

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
  BOOST_REQUIRE_GT(streamCorrect, 8550);
  BOOST_REQUIRE_GT(batchCorrect, 8550);
}

/**
 * Test majority probabilities.
 */
BOOST_AUTO_TEST_CASE(MajorityProbabilityTest)
{
  data::DatasetInfo info(1);
  StreamingDecisionTree<HoeffdingSplit<>> tree(info, 1, 3);

  // Feed the tree a few samples.
  tree.Train(arma::vec("1"), 0);
  tree.Train(arma::vec("2"), 0);
  tree.Train(arma::vec("3"), 0);

  size_t prediction;
  double probability;
  tree.Classify(arma::vec("1"), prediction, probability);

  BOOST_REQUIRE_EQUAL(prediction, 0);
  BOOST_REQUIRE_CLOSE(probability, 1.0, 1e-5);

  // Make it impure.
  tree.Train(arma::vec("4"), 1);
  tree.Classify(arma::vec("3"), prediction, probability);

  BOOST_REQUIRE_EQUAL(prediction, 0);
  BOOST_REQUIRE_CLOSE(probability, 0.75, 1e-5);

  // Flip the majority class.
  tree.Train(arma::vec("4"), 1);
  tree.Train(arma::vec("4"), 1);
  tree.Train(arma::vec("4"), 1);
  tree.Train(arma::vec("4"), 1);
  tree.Classify(arma::vec("3"), prediction, probability);

  BOOST_REQUIRE_EQUAL(prediction, 1);
  BOOST_REQUIRE_CLOSE(probability, 0.625, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
