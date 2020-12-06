/**
 * @file tests/vantage_point_tree_test.cpp
 *
 * Tests for tree-building methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::neighbor;
using namespace mlpack::metric;
using namespace mlpack::bound;

TEST_CASE("VPTreeTraitsTest", "[VantagePointTreeTest]")
{
  typedef VPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  bool b = TreeTraits<TreeType>::HasOverlappingChildren;
  REQUIRE(b == true);
  b = TreeTraits<TreeType>::FirstPointIsCentroid;
  REQUIRE(b == false);
  b = TreeTraits<TreeType>::HasSelfChildren;
  REQUIRE(b == false);
  b = TreeTraits<TreeType>::RearrangesDataset;
  REQUIRE(b == true);
  b = TreeTraits<TreeType>::BinaryTree;
  REQUIRE(b == true);
}

TEST_CASE("HollowBallBoundTest", "[VantagePointTreeTest]")
{
  HollowBallBound<EuclideanDistance> b(2, 4, arma::vec("1.0 2.0 3.0 4.0 5.0"));

  REQUIRE(b.Contains(arma::vec("1.0 2.0 3.0 7.0 5.0")) == true);

  REQUIRE(b.Contains(arma::vec("1.0 2.0 3.0 9.0 5.0")) == false);

  REQUIRE(b.Contains(arma::vec("1.0 2.0 3.0 5.0 5.0")) == false);

  HollowBallBound<EuclideanDistance> b2(0.5, 1,
      arma::vec("1.0 2.0 3.0 7.0 5.0"));
  REQUIRE(b.Contains(b2) == true);

  b2 = HollowBallBound<EuclideanDistance>(2.5, 3.5,
      arma::vec("1.0 2.0 3.0 4.5 5.0"));
  REQUIRE(b.Contains(b2) == true);

  b2 = HollowBallBound<EuclideanDistance>(2.0, 3.5,
      arma::vec("1.0 2.0 3.0 4.5 5.0"));
  REQUIRE(b.Contains(b2) == false);

  REQUIRE(b.MinDistance(arma::vec("1.0 2.0 8.0 4.0 5.0")) ==
      Approx(1.0).epsilon(1e-7));
  REQUIRE(b.MinDistance(arma::vec("1.0 2.0 4.0 4.0 5.0")) ==
      Approx(1.0).epsilon(1e-7));
  REQUIRE(b.MinDistance(arma::vec("1.0 2.0 3.0 4.0 5.0")) ==
      Approx(2.0).epsilon(1e-7));
  REQUIRE(b.MinDistance(arma::vec("1.0 2.0 5.0 4.0 5.0")) ==
      Approx(0.0).epsilon(1e-7));
  REQUIRE(b.MinDistance(arma::vec("5.0 2.0 3.0 4.0 5.0")) ==
      Approx(0.0).epsilon(1e-7));
  REQUIRE(b.MinDistance(arma::vec("3.0 2.0 3.0 4.0 5.0")) ==
      Approx(0.0).epsilon(1e-7));
  REQUIRE(b.MaxDistance(arma::vec("1.0 2.0 4.0 4.0 5.0")) ==
      Approx(5.0).epsilon(1e-7));
  REQUIRE(b.MaxDistance(arma::vec("1.0 2.0 8.0 4.0 5.0")) ==
      Approx(9.0).epsilon(1e-7));
  REQUIRE(b.MaxDistance(arma::vec("1.0 2.0 3.0 4.0 5.0")) ==
      Approx(4.0).epsilon(1e-7));

  b2 = HollowBallBound<EuclideanDistance>(3, 4,
      arma::vec("1.0 2.0 3.0 5.0 5.0"));
  REQUIRE(b.MinDistance(b2) == Approx(0.0).epsilon(1e-7));

  b2 = HollowBallBound<EuclideanDistance>(1, 2,
      arma::vec("1.0 2.0 3.0 4.0 5.0"));
  REQUIRE(b.MinDistance(b2) == Approx(0.0).epsilon(1e-7));

  b2 = HollowBallBound<EuclideanDistance>(0.5, 1.0,
      arma::vec("1.0 2.5 3.0 4.0 5.0"));
  REQUIRE(b.MinDistance(b2) == Approx(0.5).epsilon(1e-7));

  b2 = HollowBallBound<EuclideanDistance>(0.5, 1.0,
      arma::vec("1.0 8.0 3.0 4.0 5.0"));
  REQUIRE(b.MinDistance(b2) == Approx(1.0).epsilon(1e-7));

  b2 = HollowBallBound<EuclideanDistance>(0.5, 2.0,
      arma::vec("1.0 8.0 3.0 4.0 5.0"));
  REQUIRE(b.MinDistance(b2) == Approx(0.0).epsilon(1e-7));

  b2 = HollowBallBound<EuclideanDistance>(0.5, 2.0,
      arma::vec("1.0 8.0 3.0 4.0 5.0"));
  REQUIRE(b.MaxDistance(b2) == Approx(12.0).epsilon(1e-7));

  b2 = HollowBallBound<EuclideanDistance>(0.5, 2.0,
      arma::vec("1.0 3.0 3.0 4.0 5.0"));
  REQUIRE(b.MaxDistance(b2) == Approx(7.0).epsilon(1e-7));

  HollowBallBound<EuclideanDistance> b1 = b;
  b2 = HollowBallBound<EuclideanDistance>(1.0, 2.0,
      arma::vec("1.0 2.5 3.0 4.0 5.0"));

  b1 |= b2;
  REQUIRE(b1.InnerRadius() == Approx(0.5).epsilon(1e-7));

  b1 = b;
  b2 = HollowBallBound<EuclideanDistance>(0.5, 2.0,
      arma::vec("1.0 3.0 3.0 4.0 5.0"));
  b1 |= b2;
  REQUIRE(b1.InnerRadius() == Approx(0.0).epsilon(1e-7));

  b1 = b;
  b2 = HollowBallBound<EuclideanDistance>(0.5, 4.0,
      arma::vec("1.0 3.0 3.0 4.0 5.0"));
  b1 |= b2;
  REQUIRE(b1.OuterRadius() == Approx(5.0).epsilon(1e-7));
}

template<typename TreeType>
void CheckBound(TreeType& tree)
{
  typedef typename TreeType::ElemType ElemType;
  if (tree.IsLeaf())
  {
    // Ensure that the bound contains all descendant points.
    for (size_t i = 0; i < tree.NumPoints(); ++i)
    {
      ElemType dist = tree.Bound().Metric().Evaluate(tree.Bound().Center(),
          tree.Dataset().col(tree.Point(i)));
      ElemType hollowDist = tree.Bound().Metric().Evaluate(
          tree.Bound().HollowCenter(),
          tree.Dataset().col(tree.Point(i)));

      REQUIRE(tree.Bound().InnerRadius() <= hollowDist  *
          (1.0 + 10.0 * std::numeric_limits<ElemType>::epsilon()));

      REQUIRE(dist <= tree.Bound().OuterRadius() *
          (1.0 + 10.0 * std::numeric_limits<ElemType>::epsilon()));
    }
  }
  else
  {
    // Ensure that the bound contains all descendant points.
    for (size_t i = 0; i < tree.NumDescendants(); ++i)
    {
      ElemType dist = tree.Bound().Metric().Evaluate(tree.Bound().Center(),
          tree.Dataset().col(tree.Descendant(i)));
      ElemType hollowDist = tree.Bound().Metric().Evaluate(
          tree.Bound().HollowCenter(),
          tree.Dataset().col(tree.Descendant(i)));

      REQUIRE(tree.Bound().InnerRadius() <= hollowDist  *
          (1.0 + 10.0 * std::numeric_limits<ElemType>::epsilon()));

      REQUIRE(dist <= tree.Bound().OuterRadius() *
          (1.0 + 10.0 * std::numeric_limits<ElemType>::epsilon()));
    }

    CheckBound(*tree.Left());
    CheckBound(*tree.Right());
  }
}

TEST_CASE("VPTreeBoundTest", "[VantagePointTreeTest]")
{
  typedef VPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  arma::mat dataset(8, 1000);
  dataset.randu();

  TreeType tree(dataset);
  CheckBound(tree);
}

TEST_CASE("VPTreeTest", "[VantagePointTreeTest]")
{
  typedef VPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  size_t maxRuns = 10; // Ten total tests.
  size_t pointIncrements = 1000; // Range is from 2000 points to 11000.

  // We use the default leaf size of 20.
  for (size_t run = 0; run < maxRuns; run++)
  {
    size_t dimensions = run + 2;
    size_t maxPoints = (run + 1) * pointIncrements;

    size_t size = maxPoints;
    arma::mat dataset = arma::mat(dimensions, size);
    arma::mat datacopy; // Used to test mappings.

    // Mappings for post-sort verification of data.
    std::vector<size_t> newToOld;
    std::vector<size_t> oldToNew;

    // Generate data.
    dataset.randu();

    // Build the tree itself.
    TreeType root(dataset, newToOld, oldToNew);
    const arma::mat& treeset = root.Dataset();

    // Ensure the size of the tree is correct.
    REQUIRE(root.NumDescendants() == size);

    // Check the forward and backward mappings for correctness.
    for (size_t i = 0; i < size; ++i)
    {
      for (size_t j = 0; j < dimensions; ++j)
      {
        REQUIRE(treeset(j, i) == dataset(j, newToOld[i]));
        REQUIRE(treeset(j, oldToNew[i]) == dataset(j, i));
      }
    }
  }
}

TEST_CASE("SingleVPTreeTraverserTest", "[VantagePointTreeTest]")
{
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;

  // Nearest neighbor search with the VP tree.
  NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>, arma::mat,
      VPTree> knn1(dataset, SINGLE_TREE_MODE);

  knn1.Search(5, neighbors1, distances1);

  // Nearest neighbor search the naive way.
  KNN knn2(dataset, NAIVE_MODE);

  knn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); ++i)
  {
    REQUIRE(neighbors1[i] == neighbors2[i]);
    REQUIRE(distances1[i] == distances2[i]);
  }
}

TEST_CASE("DualVPTreeTraverserTest", "[VantagePointTreeTest]")
{
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;

  // Nearest neighbor search with the VP tree.
  NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>, arma::mat,
      VPTree> knn1(dataset, DUAL_TREE_MODE);

  knn1.Search(5, neighbors1, distances1);

  // Nearest neighbor search the naive way.
  KNN knn2(dataset, NAIVE_MODE);

  knn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); ++i)
  {
    REQUIRE(neighbors1[i] == neighbors2[i]);
    REQUIRE(distances1[i] == distances2[i]);
  }
}
