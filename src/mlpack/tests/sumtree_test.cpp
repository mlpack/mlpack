/**
 * @file sumtree_test.hpp
 * @author Xiaohong
 *
 * Test for Sumtree implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/reinforcement_learning/replay/sumtree.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(SumTreeTest);

/**
 * Test that we set the element.
 */
BOOST_AUTO_TEST_CASE(SetElement)
{
  SumTree<double> sumtree(4);
  sumtree.Set(0, 1.0);
  sumtree.Set(1, 0.8);
  sumtree.Set(2, 0.6);
  sumtree.Set(3, 0.4);

  BOOST_CHECK_CLOSE(sumtree.Sum(), 2.8, 1e-8);
  BOOST_CHECK_CLOSE(sumtree.Sum(0, 1), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(sumtree.Sum(0, 3), 2.4, 1e-8);
  BOOST_CHECK_CLOSE(sumtree.Sum(1, 4), 1.8, 1e-8);
}

/**
 * Test that we get the element.
 */
BOOST_AUTO_TEST_CASE(GetElement)
{
  SumTree<double> sumtree(4);
  sumtree.Set(0, 1.0);
  sumtree.Set(1, 0.8);
  sumtree.Set(2, 0.6);
  sumtree.Set(3, 0.4);

  BOOST_CHECK_CLOSE(sumtree.Get(0), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(sumtree.Get(1), 0.8, 1e-8);
  BOOST_CHECK_CLOSE(sumtree.Get(2), 0.6, 1e-8);
  BOOST_CHECK_CLOSE(sumtree.Get(3), 0.4, 1e-8);
}

/**
 * Test that we find the highest index in the array such that
 * Sum(arr[0] + arr[1] + arr[2] ... + arr[i]) <= mass.
 */
BOOST_AUTO_TEST_CASE(FindPrefixSum)
{
  SumTree<double> sumtree(4);
  sumtree.Set(0, 1.0);
  sumtree.Set(1, 0.8);
  sumtree.Set(2, 0.6);
  sumtree.Set(3, 0.4);

  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(0), 0);
  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(1), 1);
  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(2.8), 3);
  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(3.0), 3);
}

/**
 * Test that we find the highest index in the array such that
 * sum(arr[0] + arr[1] + arr[2] ... + arr[i]) <= mass.
 */
BOOST_AUTO_TEST_CASE(BatchUpdate)
{
  SumTree<double> sumtree(4);
  arma::ucolvec indices = {0, 1, 2, 3};
  arma::colvec data = {1.0, 0.8, 0.6, 0.4};

  sumtree.BatchUpdate(indices, data);

  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(0), 0);
  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(1), 1);
  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(2.8), 3);
  BOOST_CHECK_EQUAL(sumtree.FindPrefixSum(3.0), 3);
}

BOOST_AUTO_TEST_SUITE_END();
