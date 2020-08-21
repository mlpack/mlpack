/**
 * @file tests/spill_tree_test.cpp
 * @author Marcos Pividori
 *
 * Tests for the SpillTree class.  This should ensure that the class works
 * correctly and that subsequent changes don't break anything.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/tree/spill_tree.hpp>
#include "catch.hpp"
#include <stack>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::metric;

/**
 * Test to make sure the tree contains the correct number of points after
 * it is constructed. Also, it checks some invariants in the relation between
 * parent and child nodes.
 */
TEST_CASE("SpillTreeConstructionCountTest", "[SpillTreeTest]")
{
  arma::mat dataset;
  dataset.randu(3, 1000); // 1000 points in 3 dimensions.

  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  // When overlapping buffer is 0, there shouldn't be repeated points.
  TreeType tree1(dataset, 0);
  TreeType tree2 = tree1;

  REQUIRE(tree1.NumDescendants() == 1000);
  REQUIRE(tree2.NumDescendants() == 1000);

  // When overlapping buffer is greater than 0, it is possible to have repeated
  // points. So, let's check node by node, that the number of descendants
  // equals to the addition of the number of descendants of child nodes.
  TreeType tree3(dataset, 0.5);

  std::stack<TreeType*> nodes;
  nodes.push(&tree3);
  while (!nodes.empty())
  {
    TreeType* node = nodes.top();
    nodes.pop();

    size_t numDesc = node->NumPoints();

    if (node->Left())
    {
      nodes.push(node->Left());
      numDesc += node->Left()->NumDescendants();
    }

    if (node->Right())
    {
      nodes.push(node->Right());
      numDesc += node->Right()->NumDescendants();
    }

    if (node->IsLeaf())
      REQUIRE(node->NumPoints() == node->NumDescendants());
    else
      REQUIRE(node->NumPoints() == 0);

    REQUIRE(node->NumDescendants() == numDesc);
  }
}

/**
 * Test to check that parents and children are set correctly.
 */
TEST_CASE("SpillTreeConstructionParentTest", "[SpillTreeTest]")
{
  arma::mat dataset;
  dataset.randu(3, 1000); // 1000 points in 3 dimensions.

  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  TreeType tree(dataset, 0.5);

  std::stack<TreeType*> nodes;
  nodes.push(&tree);
  while (!nodes.empty())
  {
    TreeType* node = nodes.top();
    nodes.pop();

    if (node->Left())
    {
      nodes.push(node->Left());
      REQUIRE(node == node->Left()->Parent());
    }

    if (node->Right())
    {
      nodes.push(node->Right());
      REQUIRE(node == node->Right()->Parent());
    }
  }
}

/**
 * Auxiliary function to execute the same test for different flavours of Spill
 * Trees.
 */
template<typename SpillType>
void SpillTreeHyperplaneTestAux()
{
  arma::mat dataset;
  dataset.randu(3, 1000); // 1000 points in 3 dimensions.

  for (size_t cases = 0; cases < 3; cases++)
  {
    double tau = cases * 0.05;

    // Let's check node by node, that points in the left child are considered to
    // the left by the splitting hyperplane, and the same for points in the
    // right child.
    SpillType tree(dataset, tau);

    std::stack<SpillType*> nodes;
    nodes.push(&tree);
    while (!nodes.empty())
    {
      SpillType* node = nodes.top();
      nodes.pop();

      if (node->Overlap())
      {
        // We have a overlapping node.
        if (node->Left())
        {
          // Let's check that points in the left child are projected to values
          // in the range: (-inf, tau]
          size_t numDesc = node->Left()->NumDescendants();
          for (size_t i = 0; i < numDesc; ++i)
          {
            size_t descIndex = node->Left()->Descendant(i);
            REQUIRE(
                node->Hyperplane().Project(node->Dataset().col(descIndex)) <
                tau);
          }
        }
        if (node->Right())
        {
          // Let's check that points in the right child are projected to values
          // in the range: (-tau, inf)
          size_t numDesc = node->Right()->NumDescendants();
          for (size_t i = 0; i < numDesc; ++i)
          {
            size_t descIndex = node->Right()->Descendant(i);
            REQUIRE(node->Hyperplane().Project(node->Dataset().col(descIndex))
                > -tau);
          }
        }
      }
      else
      {
        // We have a non-overlapping node.
        if (node->Left())
        {
          // Let's check that points in the left child are considered to the
          // left by the splitting hyperplane.
          size_t numDesc = node->Left()->NumDescendants();
          for (size_t i = 0; i < numDesc; ++i)
          {
            size_t descIndex = node->Left()->Descendant(i);
            REQUIRE(
                node->Hyperplane().Left(node->Dataset().col(descIndex)));
          }
        }
        if (node->Right())
        {
          // Let's check that points in the right child are considered to the
          // right by the splitting hyperplane.
          size_t numDesc = node->Right()->NumDescendants();
          for (size_t i = 0; i < numDesc; ++i)
          {
            size_t descIndex = node->Right()->Descendant(i);
            REQUIRE(
                node->Hyperplane().Right(node->Dataset().col(descIndex)));
          }
        }
      }

      if (node->Left())
        nodes.push(node->Left());

      if (node->Right())
        nodes.push(node->Right());
    }
  }
}

/**
 * Test to make sure that the points in the left child are considered to the
 * left by the node's splitting hyperplane, and the same for points in the
 * right child.
 */
TEST_CASE("SpillTreeHyperplaneTest", "[SpillTreeTest]")
{
  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> SpillType1;
  typedef NonOrtSPTree<EuclideanDistance, EmptyStatistic, arma::mat> SpillType2;
  typedef MeanSPTree<EuclideanDistance, EmptyStatistic, arma::mat> SpillType3;
  typedef NonOrtMeanSPTree<EuclideanDistance, EmptyStatistic, arma::mat>
      SpillType4;

  SpillTreeHyperplaneTestAux<SpillType1>();
  SpillTreeHyperplaneTestAux<SpillType2>();
  SpillTreeHyperplaneTestAux<SpillType3>();
  SpillTreeHyperplaneTestAux<SpillType4>();
}

/**
 * Simple test for the move constructor.
 */
TEST_CASE("SpillTreeMoveConstructorTest", "[SpillTreeTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 1000);
  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  TreeType tree(dataset);

  TreeType* left = tree.Left();
  TreeType* right = tree.Right();
  size_t numDesc = tree.NumDescendants();

  TreeType newTree(std::move(tree));

  REQUIRE(tree.Left() == NULL);
  REQUIRE(tree.Right() == NULL);
  REQUIRE(tree.NumDescendants() == 0);

  REQUIRE(newTree.Left() == left);
  REQUIRE(newTree.Right() == right);
  REQUIRE(newTree.NumDescendants() == numDesc);
  if (left)
  {
    REQUIRE(newTree.Left() != NULL);
    REQUIRE(newTree.Left()->Parent() == &newTree);
  }
  if (right)
  {
    REQUIRE(newTree.Right() != NULL);
    REQUIRE(newTree.Right()->Parent() == &newTree);
  }
}

/**
 * Simple test for the copy constructor.
 */
TEST_CASE("SpillTreeCopyConstructorTest", "[SpillTreeTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 1000);
  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  TreeType* tree = new TreeType(dataset);

  TreeType* left = tree->Left();
  TreeType* right = tree->Right();
  size_t numDesc = tree->NumDescendants();

  // Copy the tree.
  TreeType newTree(*tree);

  delete tree;

  REQUIRE(newTree.Dataset().n_rows == 3);
  REQUIRE(newTree.Dataset().n_cols == 1000);
  REQUIRE(newTree.NumDescendants() == numDesc);
  if (left)
  {
    REQUIRE(newTree.Left() != left);
    REQUIRE(newTree.Left() != NULL);
    REQUIRE(newTree.Left()->Parent() == &newTree);
  }
  if (right)
  {
    REQUIRE(newTree.Right() != right);
    REQUIRE(newTree.Right() != NULL);
    REQUIRE(newTree.Right()->Parent() == &newTree);
  }
}

/**
 * Simple test for the constructor that takes a rvalue reference to the dataset.
 */
TEST_CASE("SpillTreeMoveDatasetTest", "[SpillTreeTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 1000);
  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  TreeType tree(std::move(dataset));

  REQUIRE(dataset.n_elem == 0);
  REQUIRE(tree.Dataset().n_rows == 3);
  REQUIRE(tree.Dataset().n_cols == 1000);
}
