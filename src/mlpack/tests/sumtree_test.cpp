/**
 * @file tests/sumtree_test.cpp
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

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;

/**
 * Test that we set the element.
 */
TEST_CASE("SetElement", "[SumTreeTest]")
{
  SumTree<double> sumtree(4);
  sumtree.Set(0, 1.0);
  sumtree.Set(1, 0.8);
  sumtree.Set(2, 0.6);
  sumtree.Set(3, 0.4);

  CHECK(sumtree.Sum() == Approx(2.8).epsilon(1e-10));
  CHECK(sumtree.Sum(0, 1) == Approx(1.0).epsilon(1e-10));
  CHECK(sumtree.Sum(0, 3) == Approx(2.4).epsilon(1e-10));
  CHECK(sumtree.Sum(1, 4) == Approx(1.8).epsilon(1e-10));
}

/**
 * Test that we get the element.
 */
TEST_CASE("GetElement", "[SumTreeTest]")
{
  SumTree<double> sumtree(4);
  sumtree.Set(0, 1.0);
  sumtree.Set(1, 0.8);
  sumtree.Set(2, 0.6);
  sumtree.Set(3, 0.4);

  CHECK(sumtree.Get(0) == Approx(1.0).epsilon(1e-10));
  CHECK(sumtree.Get(1) == Approx(0.8).epsilon(1e-10));
  CHECK(sumtree.Get(2) == Approx(0.6).epsilon(1e-10));
  CHECK(sumtree.Get(3) == Approx(0.4).epsilon(1e-10));
}

/**
 * Test that we find the highest index in the array such that
 * Sum(arr[0] + arr[1] + arr[2] ... + arr[i]) <= mass.
 */
TEST_CASE("FindPrefixSum", "[SumTreeTest]")
{
  SumTree<double> sumtree(4);
  sumtree.Set(0, 1.0);
  sumtree.Set(1, 0.8);
  sumtree.Set(2, 0.6);
  sumtree.Set(3, 0.4);

  CHECK(sumtree.FindPrefixSum(0) <= 0.0);
  CHECK(sumtree.FindPrefixSum(1) <= 1.0);
  CHECK(sumtree.FindPrefixSum(2.8) <= 3.0);
  CHECK(sumtree.FindPrefixSum(3.0) <= 3.0);
}

/**
 * Test that we find the highest index in the array such that
 * sum(arr[0] + arr[1] + arr[2] ... + arr[i]) <= mass.
 */
TEST_CASE("BatchUpdate", "[SumTreeTest]")
{
  SumTree<double> sumtree(4);
  arma::ucolvec indices = {0, 1, 2, 3};
  arma::colvec data = {1.0, 0.8, 0.6, 0.4};

  sumtree.BatchUpdate(indices, data);

  CHECK(sumtree.FindPrefixSum(0) <= 0);
  CHECK(sumtree.FindPrefixSum(1) <= 1);
  CHECK(sumtree.FindPrefixSum(2.8) <= 3);
  CHECK(sumtree.FindPrefixSum(3.0) <= 3);
}
