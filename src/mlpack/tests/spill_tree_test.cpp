/**
 * @file spill_tree_test.cpp
 * @author Marcos Pividori
 *
 * Tests for the SpillTree class.  This should ensure that the class works
 * correctly and that subsequent changes don't break anything.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/tree/spill_tree.hpp>
#include <boost/test/unit_test.hpp>
#include <stack>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(SpillTreeTest);

/**
 * Test to make sure the tree contains the correct number of points after
 * it is constructed. Also, it checks some invariants in the relation between
 * parent and child nodes.
 */
BOOST_AUTO_TEST_CASE(SpillTreeConstructionCountTest)
{
  arma::mat dataset;
  dataset.randu(3, 1000); // 1000 points in 3 dimensions.

  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  // When overlapping buffer is 0, there shouldn't be repeated points.
  TreeType tree1(dataset, 0);
  TreeType tree2 = tree1;

  BOOST_REQUIRE_EQUAL(tree1.NumDescendants(), 1000);
  BOOST_REQUIRE_EQUAL(tree2.NumDescendants(), 1000);

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
      BOOST_REQUIRE_EQUAL(node->NumPoints(), node->NumDescendants());
    else
      BOOST_REQUIRE_EQUAL(node->NumPoints(), 0);

    BOOST_REQUIRE_EQUAL(node->NumDescendants(), numDesc);
  }
}

/**
 * Test to check that parents and children are set correctly.
 */
BOOST_AUTO_TEST_CASE(SpillTreeConstructionParentTest)
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
      BOOST_REQUIRE_EQUAL(node, node->Left()->Parent());
    }

    if (node->Right())
    {
      nodes.push(node->Right());
      BOOST_REQUIRE_EQUAL(node, node->Right()->Parent());
    }
  }
}

/**
 * Simple test for the move constructor.
 */
BOOST_AUTO_TEST_CASE(SpillTreeMoveDatasetTest)
{
  arma::mat dataset = arma::randu<arma::mat>(3, 1000);
  typedef SPTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;

  TreeType tree(std::move(dataset));

  BOOST_REQUIRE_EQUAL(dataset.n_elem, 0);
  BOOST_REQUIRE_EQUAL(tree.Dataset().n_rows, 3);
  BOOST_REQUIRE_EQUAL(tree.Dataset().n_cols, 1000);
}

BOOST_AUTO_TEST_SUITE_END();
