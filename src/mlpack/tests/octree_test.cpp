/**
 * @file tests/octree_test.cpp
 * @author Ryan Curtin
 *
 * Test various properties of the Octree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/octree.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization_catch.hpp"

using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

/**
 * Build a quad-tree (2-d octree) on 4 points, and guarantee four points are
 * created.
 */
TEST_CASE("SimpleQuadtreeTest", "[OctreeTest]")
{
  // Four corners of the unit square.
  arma::mat dataset("0 0 1 1; 0 1 0 1");

  Octree<> t(dataset, 1);

  REQUIRE(t.NumChildren() == 4);
  REQUIRE(t.Dataset().n_cols == 4);
  REQUIRE(t.Dataset().n_rows == 2);
  REQUIRE(t.NumDescendants() == 4);
  REQUIRE(t.NumPoints() == 0);
  for (size_t i = 0; i < 4; ++i)
  {
    REQUIRE(t.Child(i).NumDescendants() == 1);
    REQUIRE(t.Child(i).NumPoints() == 1);
  }
}

/**
 * Build an octree on 3 points and make sure that only three children are
 * created.
 */
TEST_CASE("OctreeMissingChildTest", "[OctreeTest]")
{
  // Only three corners of the unit square.
  arma::mat dataset("0 0 1; 0 1 1");

  Octree<> t(dataset, 1);

  REQUIRE(t.NumChildren() == 3);
  REQUIRE(t.Dataset().n_cols == 3);
  REQUIRE(t.Dataset().n_rows == 2);
  REQUIRE(t.NumDescendants() == 3);
  REQUIRE(t.NumPoints() == 0);
  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE(t.Child(i).NumDescendants() == 1);
    REQUIRE(t.Child(i).NumPoints() == 1);
  }
}

/**
 * Ensure that building an empty octree does not fail.
 */
TEST_CASE("EmptyOctreeTest", "[OctreeTest]")
{
  arma::mat dataset;
  Octree<> t(dataset);

  REQUIRE(t.NumChildren() == 0);
  REQUIRE(t.Dataset().n_cols == 0);
  REQUIRE(t.Dataset().n_rows == 0);
  REQUIRE(t.NumDescendants() == 0);
  REQUIRE(t.NumPoints() == 0);
}

/**
 * Ensure that maxLeafSize is respected.
 */
TEST_CASE("MaxLeafSizeTest", "[OctreeTest]")
{
  arma::mat dataset(5, 15, arma::fill::randu);
  Octree<> t1(dataset, 20);
  Octree<> t2(std::move(dataset), 20);

  REQUIRE(t1.NumChildren() == 0);
  REQUIRE(t1.NumDescendants() == 15);
  REQUIRE(t1.NumPoints() == 15);

  REQUIRE(t2.NumChildren() == 0);
  REQUIRE(t2.NumDescendants() == 15);
  REQUIRE(t2.NumPoints() == 15);
}

/**
 * Check that the mappings given are correct.
 */
TEST_CASE("MappingsTest", "[OctreeTest]")
{
  // Test with both constructors.
  arma::mat dataset(3, 5, arma::fill::randu);
  arma::mat datacopy(dataset);
  std::vector<size_t> oldFromNewCopy, oldFromNewMove;

  Octree<> t1(dataset, oldFromNewCopy, 1);
  Octree<> t2(std::move(dataset), oldFromNewMove, 1);

  for (size_t i = 0; i < oldFromNewCopy.size(); ++i)
  {
    REQUIRE(arma::norm(datacopy.col(oldFromNewCopy[i]) -
        t1.Dataset().col(i)) == Approx(0.0).margin(1e-3));
    REQUIRE(arma::norm(datacopy.col(oldFromNewMove[i]) -
        t2.Dataset().col(i)) == Approx(0.0).margin(1e-3));
  }
}

/**
 * Check that the reverse mappings are correct too.
 */
TEST_CASE("ReverseMappingsTest", "[OctreeTest]")
{
  // Test with both constructors.
  arma::mat dataset(3, 300, arma::fill::randu);
  arma::mat datacopy(dataset);
  std::vector<size_t> oldFromNewCopy, oldFromNewMove, newFromOldCopy,
      newFromOldMove;

  Octree<> t1(dataset, oldFromNewCopy, newFromOldCopy);
  Octree<> t2(std::move(dataset), oldFromNewMove, newFromOldMove);

  for (size_t i = 0; i < oldFromNewCopy.size(); ++i)
  {
    REQUIRE(arma::norm(datacopy.col(oldFromNewCopy[i]) -
        t1.Dataset().col(i)) == Approx(0.0).margin(1e-3));
    REQUIRE(arma::norm(datacopy.col(oldFromNewMove[i]) -
        t2.Dataset().col(i)) == Approx(0.0).margin(1e-3));


    REQUIRE(newFromOldCopy[oldFromNewCopy[i]] == i);
    REQUIRE(newFromOldMove[oldFromNewMove[i]] == i);
  }
}

/**
 * Make sure no children at the same level are overlapping.
 */
template<typename TreeType>
void CheckOverlap(TreeType& node)
{
  // Check each combination of children.
  for (size_t i = 0; i < node.NumChildren(); ++i)
    for (size_t j = i + 1; j < node.NumChildren(); ++j)
      REQUIRE(node.Child(i).Bound().Overlap(node.Child(j).Bound()) ==
          0.0); // We need exact equality here.

  for (size_t i = 0; i < node.NumChildren(); ++i)
    CheckOverlap(node.Child(i));
}

TEST_CASE("OverlapTest", "[OctreeTest]")
{
  // Test with both constructors.
  arma::mat dataset(3, 300, arma::fill::randu);

  Octree<> t1(dataset);
  Octree<> t2(std::move(dataset));

  CheckOverlap(t1);
  CheckOverlap(t2);
}

/**
 * Make sure no points are further than the furthest point distance, and that no
 * descendants are further than the furthest descendant distance.
 */
template<typename TreeType>
void CheckFurthestDistances(TreeType& node)
{
  arma::vec center;
  node.Center(center);

  // Compare points held in the node.
  for (size_t i = 0; i < node.NumPoints(); ++i)
  {
    // Handle floating-point inaccuracies.
    REQUIRE(metric::EuclideanDistance::Evaluate(
        node.Dataset().col(node.Point(i)), center) <=
        node.FurthestPointDistance() * (1 + 1e-5));
  }

  // Compare descendants held in the node.
  for (size_t i = 0; i < node.NumDescendants(); ++i)
  {
    // Handle floating-point inaccuracies.
    REQUIRE(metric::EuclideanDistance::Evaluate(
        node.Dataset().col(node.Descendant(i)),
        center) <= node.FurthestDescendantDistance() * (1 + 1e-5));
  }

  for (size_t i = 0; i < node.NumChildren(); ++i)
    CheckFurthestDistances(node.Child(i));
}

TEST_CASE("FurthestDistanceTest", "[OctreeTest]")
{
  // Test with both constructors.
  arma::mat dataset(3, 500, arma::fill::randu);

  Octree<> t1(dataset);
  Octree<> t2(std::move(dataset));

  CheckFurthestDistances(t1);
  CheckFurthestDistances(t2);
}

/**
 * The maximum number of children a node can have is limited by the
 * dimensionality.  So we test to make sure there are no cases where we have too
 * many children.
 */
template<typename TreeType>
void CheckNumChildren(TreeType& node)
{
  REQUIRE(node.NumChildren() <= std::pow(2, node.Dataset().n_rows));
  for (size_t i = 0; i < node.NumChildren(); ++i)
    CheckNumChildren(node.Child(i));
}

TEST_CASE("MaxNumChildrenTest", "[OctreeTest]")
{
  for (size_t d = 1; d < 10; ++d)
  {
    arma::mat dataset(d, 1000 * d, arma::fill::randu);
    Octree<> t(std::move(dataset));

    CheckNumChildren(t);
  }
}

/**
 * Test the copy constructor.
 */
template<typename TreeType>
void CheckSameNode(TreeType& node1, TreeType& node2)
{
  REQUIRE(node1.NumChildren() == node2.NumChildren());
  REQUIRE(&node1.Dataset() != &node2.Dataset());

  // Make sure the children actually got copied.
  for (size_t i = 0; i < node1.NumChildren(); ++i)
    REQUIRE(&node1.Child(i) != &node2.Child(i));

  // Check that all the points are the same.
  REQUIRE(node1.NumPoints() == node2.NumPoints());
  REQUIRE(node1.NumDescendants() == node2.NumDescendants());
  for (size_t i = 0; i < node1.NumPoints(); ++i)
    REQUIRE(node1.Point(i) == node2.Point(i));
  for (size_t i = 0; i < node1.NumDescendants(); ++i)
    REQUIRE(node1.Descendant(i) == node2.Descendant(i));

  // Check that the bound is the same.
  REQUIRE(node1.Bound().Dim() == node2.Bound().Dim());
  for (size_t d = 0; d < node1.Bound().Dim(); ++d)
  {
    REQUIRE(node1.Bound()[d].Lo() ==
        Approx(node2.Bound()[d].Lo()).epsilon(1e-7));
    REQUIRE(node1.Bound()[d].Hi() ==
        Approx(node2.Bound()[d].Hi()).epsilon(1e-7));
  }

  // Check that the furthest point and descendant distance are the same.
  REQUIRE(node1.FurthestPointDistance() ==
      Approx(node2.FurthestPointDistance()).epsilon(1e-7));
  REQUIRE(node1.FurthestDescendantDistance() ==
      Approx(node2.FurthestDescendantDistance()).epsilon(1e-7));
}

TEST_CASE("CopyConstructorTest", "[OctreeTest]")
{
  // Use a small random dataset.
  arma::mat dataset(3, 100, arma::fill::randu);

  Octree<> t(dataset);
  Octree<> t2(t);

  CheckSameNode(t, t2);
}

/**
 * Test the move constructor.
 */
TEST_CASE("OcTreeTestMoveConstructorTest", "[OctreeTest]")
{
  // Use a small random dataset.
  arma::mat dataset(3, 100, arma::fill::randu);

  Octree<> t(std::move(dataset));
  Octree<> tcopy(t);

  // Move the tree.
  Octree<> t2(std::move(t));

  // Make sure the original tree has no data.
  REQUIRE(t.Dataset().n_rows == 0);
  REQUIRE(t.Dataset().n_cols == 0);
  REQUIRE(t.NumChildren() == 0);
  REQUIRE(t.NumPoints() == 0);
  REQUIRE(t.NumDescendants() == 0);
  REQUIRE(t.FurthestPointDistance() == Approx(0.0).margin(1e-5));
  REQUIRE(t.FurthestDescendantDistance() == Approx(0.0).margin(1e-5));
  REQUIRE(t.Bound().Dim() == 0);

  // Check that the new tree is the same as our copy.
  CheckSameNode(tcopy, t2);
}

/**
 * Test serialization.
 */
TEST_CASE("OctreeSerializationTest", "[OctreeTest]")
{
  // Use a small random dataset.
  arma::mat dataset(3, 500, arma::fill::randu);
  Octree<> t(std::move(dataset));

  Octree<>* xmlTree;
  Octree<>* binaryTree;
  Octree<>* textTree;

  SerializePointerObjectAll(&t, xmlTree, binaryTree, textTree);

  CheckSameNode(t, *xmlTree);
  CheckSameNode(t, *binaryTree);
  CheckSameNode(t, *textTree);

  delete xmlTree;
  delete binaryTree;
  delete textTree;
}
