/**
 * @file cosine_tree_test.cpp
 * @author Siddharth Agrawal
 *
 * Test file for CosineTree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cosine_tree/cosine_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

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
  // Initialize constants required for the test.
  const size_t numRows = 500;
  const size_t numCols = 1000;
  // Calculation accuracy.
  const double precision = 1e-15;

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
      // condition.  Due to miscalculations cosineMax calculated by
      // CosineNodeSplit may differ from cosineMax below, so we have to handle
      // minor differences.
      double cosineMax = arma::max(cosines % (cosines < 1.0 + precision));
      double cosineMin = arma::min(cosines);
      // If max(cosines) is close to 1.0 cosineMax and cosineMax2 may
      // differ significantly.
      double cosineMax2 = arma::max(cosines % (cosines < 1.0 - precision));


      if (std::fabs(cosineMax - cosineMax2) < precision)
      {
        // Check with some precision.
        for (i = 0; i < leftIndices.size(); i++)
          BOOST_REQUIRE_LT(cosineMax - cosines(i),
                           cosines(i) - cosineMin + precision);

        for (j = 0, k = i; j < rightIndices.size(); j++, k++)
          BOOST_REQUIRE_GT(cosineMax - cosines(k),
                           cosines(k) - cosineMin - precision);
      }
      else
      {
        size_t numMax1Errors = 0;
        size_t numMax2Errors = 0;

        // Find errors for cosineMax.
        for (i = 0; i < leftIndices.size(); i++)
          if (cosineMax - cosines(i) >= cosines(i) - cosineMin + precision)
            numMax1Errors++;

        for (j = 0, k = i; j < rightIndices.size(); j++, k++)
          if (cosineMax - cosines(k) <= cosines(k) - cosineMin - precision)
            numMax1Errors++;

        // Find errors for cosineMax2.
        for (i = 0; i < leftIndices.size(); i++)
          if (cosineMax2 - cosines(i) >= cosines(i) - cosineMin + precision)
            numMax2Errors++;

        for (j = 0, k = i; j < rightIndices.size(); j++, k++)
          if (cosineMax2 - cosines(k) <= cosines(k) - cosineMin - precision)
            numMax2Errors++;

        // One of the maximum cosine values should be correct
        BOOST_REQUIRE_EQUAL(std::min(numMax1Errors, numMax2Errors), 0);
      }
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
