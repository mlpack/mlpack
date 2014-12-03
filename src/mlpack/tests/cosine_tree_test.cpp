/**
 * @file cosine_tree_test.cpp
 * @author Siddharth Agrawal
 *
 * Test file for CosineTree class.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cosine_tree/cosine_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(CosineTreeTest);

using namespace mlpack;
using namespace mlpack::tree;

/**
 * Constructs a cosine tree with epsilon = 1. Checks if the root node is split
 * further, as it shouldn't be.
 */
BOOST_AUTO_TEST_CASE(CosineTreeNoSplit)
{
  // Initialize constants required for the test.
  const size_t numRows = 10;
  const size_t numCols = 15;
  const double epsilon = 1;
  const double delta = 0.1;

  // Make a random dataset.
  arma::mat data = arma::randu(numRows, numCols);

  // Make a cosine tree, with the generated dataset and the defined constants.
  // Note that the value of epsilon is one.
  CosineTree ctree(data, epsilon, delta);
  arma::mat basis;
  ctree.GetFinalBasis(basis);

  // Since epsilon is one, there should be no splitting and the only vector in
  // the basis should come from the root node.
  BOOST_REQUIRE_EQUAL(basis.n_cols, 1);
}

/**
 * Checks CosineTree::CosineNodeSplit() by doing a depth first search on a
 * random dataset and checking if it satisfies the split condition.
 */
BOOST_AUTO_TEST_CASE(CosineNodeCosineSplit)
{
  // Intialize constants required for the test.
  const size_t numRows = 500;
  const size_t numCols = 1000;

  // Make a random dataset and the root object.
  arma::mat data = arma::randu(numRows, numCols);
  CosineTree root(data);

  // Stack for depth first search of the tree.
  std::vector<CosineTree*> nodeStack;
  nodeStack.push_back(&root);

  // While stack is not empty.
  while (nodeStack.size())
  {
    // Pop a node from the stack and split it.
    CosineTree *currentNode, *currentLeft, *currentRight;
    currentNode = nodeStack.back();
    currentNode->CosineNodeSplit();
    nodeStack.pop_back();

    // Obtain pointers to the children of the node.
    currentLeft = currentNode->Left();
    currentRight = currentNode->Right();

    // If children exist.
    if (currentLeft && currentRight)
    {
      // Push the child nodes on to the stack.
      nodeStack.push_back(currentLeft);
      nodeStack.push_back(currentRight);

      // Obtain the split point of the popped node.
      arma::vec splitPoint = data.col(currentNode->SplitPointIndex());

      // Column indices of the the child nodes.
      std::vector<size_t> leftIndices, rightIndices;
      leftIndices = currentLeft->VectorIndices();
      rightIndices = currentRight->VectorIndices();

      // The columns in the popped should be split into left and right nodes.
      BOOST_REQUIRE_EQUAL(currentNode->NumColumns(), leftIndices.size() +
          rightIndices.size());

      // Calculate the cosine values for each of the columns in the node.
      arma::vec cosines;
      cosines.zeros(currentNode->NumColumns());

      size_t i, j, k;
      for (i = 0; i < leftIndices.size(); i++)
        cosines(i) = arma::norm_dot(data.col(leftIndices[i]), splitPoint);

      for (j = 0, k = i; j < rightIndices.size(); j++, k++)
        cosines(k) = arma::norm_dot(data.col(rightIndices[j]), splitPoint);

      // Check if the columns assigned to the children agree with the splitting
      // condition.
      double cosineMax = arma::max(cosines % (cosines < 1));
      double cosineMin = arma::min(cosines);

      for (i = 0; i < leftIndices.size(); i++)
        BOOST_CHECK_LT(cosineMax - cosines(i), cosines(i) - cosineMin);

      for (j = 0, k = i; j < rightIndices.size(); j++, k++)
        BOOST_CHECK_GT(cosineMax - cosines(k), cosines(k) - cosineMin);
    }
  }
}

/**
 * Checks CosineTree::ModifiedGramSchmidt() by creating a random basis for the
 * vector subspace and checking if all the vectors are orthogonal to each other.
 */
BOOST_AUTO_TEST_CASE(CosineTreeModifiedGramSchmidt)
{
  // Initialize constants required for the test.
  const size_t numRows = 100;
  const size_t numCols = 50;
  const double epsilon = 1;
  const double delta = 0.1;

  // Make a random dataset.
  arma::mat data = arma::randu(numRows, numCols);

  // Declare a queue and a dummy CosineTree object.
  CosineNodeQueue basisQueue;
  CosineTree dummyTree(data, epsilon, delta);

  for(size_t i = 0; i < numCols; i++)
  {
    // Make a new CosineNode object.
    CosineTree* basisNode;
    basisNode = new CosineTree(data);

    // Use the columns of the dataset as random centroids.
    arma::vec centroid = data.col(i);
    arma::vec newBasisVector;

    // Obtain the orthonormalized version of the centroid.
    dummyTree.ModifiedGramSchmidt(basisQueue, centroid, newBasisVector);

    // Check if the obtained vector is orthonormal to the basis vectors.
    CosineNodeQueue::const_iterator j = basisQueue.begin();
    CosineTree* currentNode;

    for(; j != basisQueue.end(); j++)
    {
      currentNode = *j;
      BOOST_REQUIRE_SMALL(arma::dot(currentNode->BasisVector(), newBasisVector),
                          1e-5);
    }

    // Add the obtained vector to the basis.
    basisNode->BasisVector(newBasisVector);
    basisNode->L2Error(arma::randu());
    basisQueue.push(basisNode);
  }

  // Deallocate memory given to the objects.
  for(size_t i = 0; i < numCols; i++)
  {
    CosineTree* currentNode;
    currentNode = basisQueue.top();
    basisQueue.pop();

    delete currentNode;
  }
}

BOOST_AUTO_TEST_SUITE_END();
