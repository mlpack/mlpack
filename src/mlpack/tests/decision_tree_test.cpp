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
- aux split info is empty
- basic construction test
- build on sparse test on dense
- efficacy test

*/
BOOST_AUTO_TEST_SUITE_END();
