/**
 * @file ub_tree_test.cpp
 * @author Mikhail Lozhnikov
 *
 * Tests for the UB tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;
using namespace mlpack::neighbor;

BOOST_AUTO_TEST_SUITE(UBTreeTest);

BOOST_AUTO_TEST_CASE(AddressTest)
{
  typedef double ElemType;
  typedef typename std::conditional<sizeof(ElemType) * CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type AddressElemType;
  arma::Mat<ElemType> dataset(8, 1000);

  dataset.randu();
  dataset -= 0.5;
  arma::Col<AddressElemType> address(dataset.n_rows);
  arma::Col<ElemType> point(dataset.n_rows);

  // Ensure that this is one-to-one transform.
  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    addr::PointToAddress(address, dataset.col(i));
    addr::AddressToPoint(point, address);

    for (size_t k = 0; k < dataset.n_rows; k++)
      BOOST_REQUIRE_CLOSE(dataset(k, i), point[k], 1e-13);
  }

}

template<typename TreeType>
void CheckSplit(const TreeType& tree)
{
  typedef typename TreeType::ElemType ElemType;
  typedef typename std::conditional<sizeof(ElemType) * CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type AddressElemType;

  if (tree.IsLeaf())
    return;

  arma::Col<AddressElemType> lo(tree.Bound().Dim());
  arma::Col<AddressElemType> hi(tree.Bound().Dim());

  lo.fill(std::numeric_limits<AddressElemType>::max());
  hi.fill(0);

  arma::Col<AddressElemType> address(tree.Bound().Dim());

  // Find the highest address of the left node.
  for (size_t i = 0; i < tree.Left()->NumDescendants(); i++)
  {
    addr::PointToAddress(address,
        tree.Dataset().col(tree.Left()->Descendant(i)));

    if (addr::CompareAddresses(address, hi) > 0)
      hi = address;
  }

  // Find the lowest address of the right node.
  for (size_t i = 0; i < tree.Right()->NumDescendants(); i++)
  {
    addr::PointToAddress(address,
        tree.Dataset().col(tree.Right()->Descendant(i)));

    if (addr::CompareAddresses(address, lo) < 0)
      lo = address;
  }

  // Addresses in the left node should be less than addresses in the right node.
  BOOST_REQUIRE_LE(addr::CompareAddresses(hi, lo), 0);

  CheckSplit(*tree.Left());
  CheckSplit(*tree.Right());
}

BOOST_AUTO_TEST_CASE(UBTreeSplitTest)
{
  typedef UBTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  arma::mat dataset(8, 1000);

  dataset.randu();

  TreeType tree(dataset);
  CheckSplit(tree);
}

template<typename TreeType>
void CheckBound(const TreeType& tree)
{
  typedef typename TreeType::ElemType ElemType;
  for (size_t i = 0; i < tree.NumDescendants(); i++)
  {
    arma::Col<ElemType> point = tree.Dataset().col(tree.Descendant(i));

    // Check that the point is contained in the bound.
    BOOST_REQUIRE_EQUAL(true, tree.Bound().Contains(point));

    const arma::Mat<ElemType>& loBound = tree.Bound().LoBound();
    const arma::Mat<ElemType>& hiBound = tree.Bound().HiBound();

    // Ensure that there is a hyperrectangle that contains the point.
    bool success = false;
    for (size_t j = 0; j < tree.Bound().NumBounds(); j++)
    {
      success = true;
      for (size_t k = 0; k < loBound.n_rows; k++)
      {
        if (point[k] < loBound(k, j) - 1e-14 * std::fabs(loBound(k, j)) ||
            point[k] > hiBound(k, j) + 1e-14 * std::fabs(hiBound(k, j)))
        {
          success = false;
          break;
        }
      }
      if (success)
        break;
    }

    BOOST_REQUIRE_EQUAL(success, true);
  }

  if (!tree.IsLeaf())
  {
    CheckBound(*tree.Left());
    CheckBound(*tree.Right());
  }
}

BOOST_AUTO_TEST_CASE(UBTreeBoundTest)
{
  typedef UBTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  arma::mat dataset(8, 1000);

  dataset.randu();

  TreeType tree(dataset);
  CheckBound(tree);
}

// Ensure that MinDistance() and MaxDistance() works correctly.
template<typename TreeType, typename MetricType>
void CheckDistance(TreeType& tree, TreeType* node = NULL)
{
  typedef typename TreeType::ElemType ElemType;
  if (node == NULL)
  {
    node = &tree;

    while (node->Parent() != NULL)
      node = node->Parent();

    CheckDistance<TreeType, MetricType>(tree, node);

    for (size_t j = 0; j < tree.Dataset().n_cols; j++)
    {
      const arma::Col<ElemType>& point = tree.  Dataset().col(j);
      ElemType maxDist = 0;
      ElemType minDist = std::numeric_limits<ElemType>::max();
      for (size_t i = 0; i < tree.NumDescendants(); i++)
      {
        ElemType dist = MetricType::Evaluate(
            tree.Dataset().col(tree.Descendant(i)),
            tree.Dataset().col(j));

        if (dist > maxDist)
          maxDist = dist;
        if (dist < minDist)
          minDist = dist;
      }

      BOOST_REQUIRE_LE(tree.Bound().MinDistance(point), minDist *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));
      BOOST_REQUIRE_LE(maxDist, tree.Bound().MaxDistance(point) *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));

      math::RangeType<ElemType> r = tree.Bound().RangeDistance(point);

      BOOST_REQUIRE_LE(r.Lo(), minDist *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));
      BOOST_REQUIRE_LE(maxDist, r.Hi() *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));
    }
      
    if (!tree.IsLeaf())
    {
      CheckDistance<TreeType, MetricType>(*tree.Left());
      CheckDistance<TreeType, MetricType>(*tree.Right());
    }
  }
  else
  {
    if (&tree != node)
    {
      ElemType maxDist = 0;
      ElemType minDist = std::numeric_limits<ElemType>::max();
      for (size_t i = 0; i < tree.NumDescendants(); i++)
        for (size_t j = 0; j < node->NumDescendants(); j++)
        {
          ElemType dist = MetricType::Evaluate(
              tree.Dataset().col(tree.Descendant(i)),
              node->Dataset().col(node->Descendant(j)));

          if (dist > maxDist)
            maxDist = dist;
          if (dist < minDist)
            minDist = dist;
        }

      BOOST_REQUIRE_LE(tree.Bound().MinDistance(node->Bound()), minDist *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));
      BOOST_REQUIRE_LE(maxDist, tree.Bound().MaxDistance(node->Bound()) *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));

      math::RangeType<ElemType> r = tree.Bound().RangeDistance(node->Bound());

      BOOST_REQUIRE_LE(r.Lo(), minDist *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));
      BOOST_REQUIRE_LE(maxDist, r.Hi() *
          (1.0 + 10 * std::numeric_limits<ElemType>::epsilon()));
    }
    if (!node->IsLeaf())
    {
      CheckDistance<TreeType, MetricType>(tree, node->Left());
      CheckDistance<TreeType, MetricType>(tree, node->Right());
    }
  }
}

BOOST_AUTO_TEST_CASE(UBTreeDistanceTest)
{
  typedef UBTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  arma::mat dataset(8, 1000);

  dataset.randu();

  TreeType tree(dataset);
  CheckDistance<TreeType, EuclideanDistance>(tree);
}


BOOST_AUTO_TEST_CASE(UBTreeTest)
{
  typedef UBTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

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
    BOOST_REQUIRE_EQUAL(root.NumDescendants(), size);

    // Check the forward and backward mappings for correctness.
    for (size_t i = 0; i < size; i++)
    {
      for (size_t j = 0; j < dimensions; j++)
      {
        BOOST_REQUIRE_EQUAL(treeset(j, i), dataset(j, newToOld[i]));
        BOOST_REQUIRE_EQUAL(treeset(j, oldToNew[i]), dataset(j, i));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(SingleTreeTraverserTest)
{
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;

  // Nearest neighbor search with the UB tree.
  NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>, arma::mat,
      UBTree> knn1(dataset, SINGLE_TREE_MODE);

  knn1.Search(5, neighbors1, distances1);

  // Nearest neighbor search the naive way.
  KNN knn2(dataset, NAIVE_MODE);

  knn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); i++)
  {
    BOOST_REQUIRE_EQUAL(neighbors1[i], neighbors2[i]);
    BOOST_REQUIRE_EQUAL(distances1[i], distances2[i]);
  }
}

BOOST_AUTO_TEST_CASE(DualTreeTraverserTest)
{
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;

  // Nearest neighbor search with the UB tree.
  NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>, arma::mat,
      UBTree> knn1(dataset, DUAL_TREE_MODE);

  knn1.Search(5, neighbors1, distances1);

  // Nearest neighbor search the naive way.
  KNN knn2(dataset, NAIVE_MODE);

  knn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); i++)
  {
    BOOST_REQUIRE_EQUAL(neighbors1[i], neighbors2[i]);
    BOOST_REQUIRE_EQUAL(distances1[i], distances2[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END();
