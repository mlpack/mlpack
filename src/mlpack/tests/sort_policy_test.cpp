/**
 * @file tests/sort_policy_test.cpp
 * @author Ryan Curtin
 *
 * Tests for each of the implementations of the SortPolicy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

// Classes to test.
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::bound;
using namespace mlpack::tree;
using namespace mlpack::metric;

// Tests for NearestNeighborSort

/**
 * Ensure the best distance for nearest neighbors is 0.
 */
TEST_CASE("NnsBestDistance", "[SortPolicyTest]")
{
  REQUIRE(NearestNeighborSort::BestDistance() == 0);
}

/**
 * Ensure the worst distance for nearest neighbors is DBL_MAX.
 */
TEST_CASE("NnsWorstDistance", "[SortPolicyTest]")
{
  REQUIRE(NearestNeighborSort::WorstDistance() == DBL_MAX);
}

/**
 * Make sure the comparison works for values strictly less than the reference.
 */
TEST_CASE("NnsIsBetterStrict", "[SortPolicyTest]")
{
  REQUIRE(NearestNeighborSort::IsBetter(5.0, 6.0) == true);
}

/**
 * Warn in case the comparison is not strict.
 */
TEST_CASE("NnsIsBetterNotStrict", "[SortPolicyTest]")
{
  CHECK(NearestNeighborSort::IsBetter(6.0, 6.0) == true);
}

/**
 * Very simple sanity check to ensure that bounds are working alright.  We will
 * use a one-dimensional bound for simplicity.
 */
TEST_CASE("NnsNodeToNodeDistance", "[SortPolicyTest]")
{
  // Well, there's no easy way to make HRectBounds the way we want, so we have
  // to make them and then expand the region to include new points.
  arma::mat dataset("1");
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType nodeOne(dataset);
  arma::vec utility(1);
  utility[0] = 0;

  nodeOne.Bound() = HRectBound<EuclideanDistance>(1);
  nodeOne.Bound() |= utility;
  utility[0] = 1;
  nodeOne.Bound() |= utility;

  TreeType nodeTwo(dataset);
  nodeTwo.Bound() = HRectBound<EuclideanDistance>(1);

  utility[0] = 5;
  nodeTwo.Bound() |= utility;
  utility[0] = 6;
  nodeTwo.Bound() |= utility;

  // This should use the L2 distance.
  REQUIRE(NearestNeighborSort::BestNodeToNodeDistance(&nodeOne, &nodeTwo) ==
      Approx(4.0).epsilon(1e-7));

  // And another just to be sure, from the other side.
  nodeTwo.Bound().Clear();
  utility[0] = -2;
  nodeTwo.Bound() |= utility;
  utility[0] = -1;
  nodeTwo.Bound() |= utility;

  // Again, the distance is the L2 distance.
  REQUIRE(NearestNeighborSort::BestNodeToNodeDistance(&nodeOne, &nodeTwo) ==
      Approx(1.0).epsilon(1e-7));

  // Now, when the bounds overlap.
  nodeTwo.Bound().Clear();
  utility[0] = -0.5;
  nodeTwo.Bound() |= utility;
  utility[0] = 0.5;
  nodeTwo.Bound() |= utility;

  REQUIRE(NearestNeighborSort::BestNodeToNodeDistance(&nodeOne, &nodeTwo) == 
      Approx(0.0).margin(1e-5));
}

/**
 * Another very simple sanity check for the point-to-node case, again in one
 * dimension.
 */
TEST_CASE("NnsPointToNodeDistance", "[SortPolicyTest]")
{
  // Well, there's no easy way to make HRectBounds the way we want, so we have
  // to make them and then expand the region to include new points.
  arma::vec utility(1);
  utility[0] = 0;

  arma::mat dataset("1");
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType node(dataset);
  node.Bound() = HRectBound<EuclideanDistance>(1);
  node.Bound() |= utility;
  utility[0] = 1;
  node.Bound() |= utility;

  arma::vec point(1);
  point[0] = -0.5;

  // The distance is the L2 distance.
  REQUIRE(NearestNeighborSort::BestPointToNodeDistance(point, &node) ==
      Approx(0.5).epsilon(1e-7));

  // Now from the other side of the bound.
  point[0] = 1.5;

  REQUIRE(NearestNeighborSort::BestPointToNodeDistance(point, &node) ==
      Approx(0.5).epsilon(1e-7));

  // And now when the point is inside the bound.
  point[0] = 0.5;

  REQUIRE(NearestNeighborSort::BestPointToNodeDistance(point, &node) == 
      Approx(0.0).margin(1e-5));
}

// Tests for FurthestNeighborSort

/**
 * Ensure the best distance for furthest neighbors is DBL_MAX.
 */
TEST_CASE("FnsBestDistance", "[SortPolicyTest]")
{
  REQUIRE(FurthestNeighborSort::BestDistance() == DBL_MAX);
}

/**
 * Ensure the worst distance for furthest neighbors is 0.
 */
TEST_CASE("FnsWorstDistance", "[SortPolicyTest]")
{
  REQUIRE(FurthestNeighborSort::WorstDistance() == 0);
}

/**
 * Make sure the comparison works for values strictly less than the reference.
 */
TEST_CASE("FnsIsBetterStrict", "[SortPolicyTest]")
{
  REQUIRE(FurthestNeighborSort::IsBetter(5.0, 4.0) == true);
}

/**
 * Warn in case the comparison is not strict.
 */
TEST_CASE("FnsIsBetterNotStrict", "[SortPolicyTest]")
{
  CHECK(FurthestNeighborSort::IsBetter(6.0, 6.0) == true);
}

/**
 * Very simple sanity check to ensure that bounds are working alright.  We will
 * use a one-dimensional bound for simplicity.
 */
TEST_CASE("FnsNodeToNodeDistance", "[SortPolicyTest]")
{
  // Well, there's no easy way to make HRectBounds the way we want, so we have
  // to make them and then expand the region to include new points.
  arma::vec utility(1);
  utility[0] = 0;

  arma::mat dataset("1");
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType nodeOne(dataset);
  nodeOne.Bound() = HRectBound<EuclideanDistance>(1);
  nodeOne.Bound() |= utility;
  utility[0] = 1;
  nodeOne.Bound() |= utility;

  TreeType nodeTwo(dataset);
  nodeTwo.Bound() = HRectBound<EuclideanDistance>(1);
  utility[0] = 5;
  nodeTwo.Bound() |= utility;
  utility[0] = 6;
  nodeTwo.Bound() |= utility;

  // This should use the L2 distance.
  REQUIRE(FurthestNeighborSort::BestNodeToNodeDistance(&nodeOne, &nodeTwo) ==
      Approx(6.0).epsilon(1e-7));

  // And another just to be sure, from the other side.
  nodeTwo.Bound().Clear();
  utility[0] = -2;
  nodeTwo.Bound() |= utility;
  utility[0] = -1;
  nodeTwo.Bound() |= utility;

  // Again, the distance is the L2 distance.
  REQUIRE(FurthestNeighborSort::BestNodeToNodeDistance(&nodeOne, &nodeTwo) ==
      Approx(3.0).epsilon(1e-7));

  // Now, when the bounds overlap.
  nodeTwo.Bound().Clear();
  utility[0] = -0.5;
  nodeTwo.Bound() |= utility;
  utility[0] = 0.5;
  nodeTwo.Bound() |= utility;

  REQUIRE(FurthestNeighborSort::BestNodeToNodeDistance(&nodeOne, &nodeTwo) ==
      Approx(1.5).epsilon(1e-7));
}

/**
 * Another very simple sanity check for the point-to-node case, again in one
 * dimension.
 */
TEST_CASE("FnsPointToNodeDistance", "[SortPolicyTest]")
{
  // Well, there's no easy way to make HRectBounds the way we want, so we have
  // to make them and then expand the region to include new points.
  arma::vec utility(1);
  utility[0] = 0;

  arma::mat dataset("1");
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType node(dataset);
  node.Bound() = HRectBound<EuclideanDistance>(1);
  node.Bound() |= utility;
  utility[0] = 1;
  node.Bound() |= utility;

  arma::vec point(1);
  point[0] = -0.5;

  // The distance is the L2 distance.
  REQUIRE(FurthestNeighborSort::BestPointToNodeDistance(point, &node) ==
      Approx(1.5).epsilon(1e-7));

  // Now from the other side of the bound.
  point[0] = 1.5;

  REQUIRE(FurthestNeighborSort::BestPointToNodeDistance(point, &node) ==
      Approx(1.5).epsilon(1e-7));

  // And now when the point is inside the bound.
  point[0] = 0.5;

  REQUIRE(FurthestNeighborSort::BestPointToNodeDistance(point, &node) ==
      Approx(0.5).epsilon(1e-7));
}
