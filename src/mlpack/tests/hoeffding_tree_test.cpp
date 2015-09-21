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

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
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
  arma::Mat<size_t> counts(4, 3);

  counts(0, 0) = 0;
  counts(0, 1) = 0;
  counts(0, 2) = 10;

  counts(1, 0) = 5;
  counts(1, 1) = 5;
  counts(1, 2) = 0;

  counts(2, 0) = 4;
  counts(2, 1) = 4;
  counts(2, 2) = 4;

  counts(3, 0) = 8;
  counts(3, 1) = 1;
  counts(3, 2) = 1;

  // The Gini impurity of the whole thing is ... not yet calculated.
}

BOOST_AUTO_TEST_CASE(GiniImpurityZeroTest)
{
  // When nothing has been seen, the gini impurity should be zero.
  arma::Mat<size_t> counts = arma::zeros<arma::Mat<size_t>>(10, 10);

  BOOST_REQUIRE_SMALL(GiniImpurity::Evaluate(counts), 1e-10);
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
    split.Train(math::RandInt(0, 10), 1);
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

  split.Train(math::RandInt(0, 10), 1);
  for (size_t i = 0; i < 250; ++i)
  {
    split.Train(math::RandInt(0, 10), 1);
    split.Train(math::RandInt(0, 10), 2);
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
  std::vector<StreamingDecisionTree<HoeffdingSplit>> children;
  data::DatasetInfo info;
  info.MapString("hello", 0); // Make dimension 0 categorical.
  HoeffdingCategoricalSplit<GiniImpurity>::SplitInfo splitInfo;

  // Create the children.
  split.CreateChildren(children, info, splitInfo);

  BOOST_REQUIRE_EQUAL(children.size(), 3);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(0), 0);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(1), 1);
  BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(2), 2);
}

BOOST_AUTO_TEST_SUITE_END();
