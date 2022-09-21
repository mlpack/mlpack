/**
 * @file tests/serialization_test.cpp
 * @author Ryan Curtin
 *
 * Test serialization of mlpack objects.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "catch.hpp"
#include "serialization.hpp"

#include <mlpack/methods/ann.hpp>
#include <mlpack/methods/hoeffding_trees.hpp>
#include <mlpack/methods/perceptron.hpp>
#include <mlpack/methods/logistic_regression.hpp>
#include <mlpack/methods/neighbor_search.hpp>
#include <mlpack/methods/softmax_regression.hpp>
#include <mlpack/methods/det.hpp>
#include <mlpack/methods/naive_bayes.hpp>
#include <mlpack/methods/rann.hpp>
#include <mlpack/methods/lsh.hpp>
#include <mlpack/methods/lars.hpp>
#include <mlpack/methods/bayesian_linear_regression.hpp>

using namespace mlpack;

using namespace arma;
using namespace cereal;
using namespace std;

/**
 * Serialize a random cube.
 */
TEST_CASE("CubeSerializeTest", "[SerializationTest]")
{
  arma::cube m;
  m.randu(2, 50, 50);
  TestAllArmadilloSerialization(m);
}

/**
 * Serialize an empty cube.
 */
TEST_CASE("EmptyCubeSerializeTest", "[SerializationTest]")
{
  arma::cube c;
  TestAllArmadilloSerialization(c);
}


/**
 * Can we load and save an Armadillo matrix?
 */
TEST_CASE("MatrixSerializeXMLTest", "[SerializationTest]")
{
  arma::mat m;
  m.randu(50, 50);
  TestAllArmadilloSerialization(m);
}

/**
 * How about columns?
 */
TEST_CASE("ColSerializeTest", "[SerializationTest]")
{
  arma::vec m;
  m.randu(50, 1);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
TEST_CASE("RowSerializeTest", "[SerializationTest]")
{
  arma::rowvec m;
  m.randu(1, 50);
  TestAllArmadilloSerialization(m);
}

// A quick test with an empty matrix.
TEST_CASE("EmptyMatrixSerializeTest", "[SerializationTest]")
{
  arma::mat m;
  TestAllArmadilloSerialization(m);
}

/**
 * Can we load and save a sparse Armadillo matrix?
 */
TEST_CASE("SparseMatrixSerializeTest", "[SerializationTest]")
{
  arma::sp_mat m;
  m.sprandu(50, 50, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about columns?
 */
TEST_CASE("SparseColSerializeTest", "[SerializationTest]")
{
  arma::sp_vec m;
  m.sprandu(50, 1, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
TEST_CASE("SparseRowSerializeTest", "[SerializationTest]")
{
  arma::sp_rowvec m;
  m.sprandu(1, 50, 0.3);
  TestAllArmadilloSerialization(m);
}

// A quick test with an empty matrix.
TEST_CASE("EmptySparseMatrixSerializeTest", "[SerializationTest]")
{
  arma::sp_mat m;
  TestAllArmadilloSerialization(m);
}

TEST_CASE("BallBoundTest", "[SerializationTest]")
{
  BallBound<> b(100);
  b.Center().randu();
  b.Radius() = 14.0;

  BallBound<> xmlB, jsonB, binaryB;

  SerializeObjectAll(b, xmlB, jsonB, binaryB);

  // Check the dimensionality.
  REQUIRE(b.Dim() == xmlB.Dim());
  REQUIRE(b.Dim() == jsonB.Dim());
  REQUIRE(b.Dim() == binaryB.Dim());

  // Check the radius.
  REQUIRE(b.Radius() == Approx(xmlB.Radius()).epsilon(1e-10));
  REQUIRE(b.Radius() == Approx(jsonB.Radius()).epsilon(1e-10));
  REQUIRE(b.Radius() == Approx(binaryB.Radius()).epsilon(1e-10));

  // Now check the vectors.
  CheckMatrices(b.Center(), xmlB.Center(), jsonB.Center(), binaryB.Center());
}

TEST_CASE("MahalanobisBallBoundTest", "[SerializationTest]")
{
  BallBound<MahalanobisDistance<>, arma::vec> b(100);
  b.Center().randu();
  b.Radius() = 14.0;
  b.Metric().Covariance().randu(100, 100);

  BallBound<MahalanobisDistance<>, arma::vec> xmlB, jsonB, binaryB;

  SerializeObjectAll(b, xmlB, jsonB, binaryB);

  // Check the radius.
  REQUIRE(b.Radius() == Approx(xmlB.Radius()).epsilon(1e-10));
  REQUIRE(b.Radius() == Approx(jsonB.Radius()).epsilon(1e-10));
  REQUIRE(b.Radius() == Approx(binaryB.Radius()).epsilon(1e-10));

  // Check the vectors.
  CheckMatrices(b.Center(), xmlB.Center(), jsonB.Center(), binaryB.Center());
  CheckMatrices(b.Metric().Covariance(),
                xmlB.Metric().Covariance(),
                jsonB.Metric().Covariance(),
                binaryB.Metric().Covariance());
}

TEST_CASE("HRectBoundTest", "[SerializationTest]")
{
  HRectBound<> b(2);

  arma::mat points("0.0, 1.1; 5.0, 2.2");
  points = points.t();
  b |= points; // [0.0, 5.0]; [1.1, 2.2];

  HRectBound<> xmlB, jsonB, binaryB;

  SerializeObjectAll(b, xmlB, jsonB, binaryB);

  // Check the dimensionality.
  REQUIRE(b.Dim() == xmlB.Dim());
  REQUIRE(b.Dim() == jsonB.Dim());
  REQUIRE(b.Dim() == binaryB.Dim());

  // Check the bounds.
  for (size_t i = 0; i < b.Dim(); ++i)
  {
    REQUIRE(b[i].Lo() == Approx(xmlB[i].Lo()).epsilon(1e-10));
    REQUIRE(b[i].Hi() == Approx(xmlB[i].Hi()).epsilon(1e-10));
    REQUIRE(b[i].Lo() == Approx(jsonB[i].Lo()).epsilon(1e-10));
    REQUIRE(b[i].Hi() == Approx(jsonB[i].Hi()).epsilon(1e-10));
    REQUIRE(b[i].Lo() == Approx(binaryB[i].Lo()).epsilon(1e-10));
    REQUIRE(b[i].Hi() == Approx(binaryB[i].Hi()).epsilon(1e-10));
  }

  // Check the minimum width.
  REQUIRE(b.MinWidth() == Approx(xmlB.MinWidth()).epsilon(1e-10));
  REQUIRE(b.MinWidth() == Approx(jsonB.MinWidth()).epsilon(1e-10));
  REQUIRE(b.MinWidth() == Approx(binaryB.MinWidth()).epsilon(1e-10));
}

template<typename TreeType>
void CheckTrees(TreeType& tree,
                TreeType& xmlTree,
                TreeType& jsonTree,
                TreeType& binaryTree)
{
  const typename TreeType::Mat* dataset = &tree.Dataset();

  // Make sure that the data matrices are the same.
  if (tree.Parent() == NULL)
  {
    CheckMatrices(*dataset,
                  xmlTree.Dataset(),
                  jsonTree.Dataset(),
                  binaryTree.Dataset());

    // Also ensure that the other parents are null too.
    REQUIRE(xmlTree.Parent() == (TreeType*) NULL);
    REQUIRE(jsonTree.Parent() == (TreeType*) NULL);
    REQUIRE(binaryTree.Parent() == (TreeType*) NULL);
  }

  // Make sure the number of children is the same.
  REQUIRE(tree.NumChildren() == xmlTree.NumChildren());
  REQUIRE(tree.NumChildren() == jsonTree.NumChildren());
  REQUIRE(tree.NumChildren() == binaryTree.NumChildren());

  // Make sure the number of descendants is the same.
  REQUIRE(tree.NumDescendants() == xmlTree.NumDescendants());
  REQUIRE(tree.NumDescendants() == jsonTree.NumDescendants());
  REQUIRE(tree.NumDescendants() == binaryTree.NumDescendants());

  // Make sure the number of points is the same.
  REQUIRE(tree.NumPoints() == xmlTree.NumPoints());
  REQUIRE(tree.NumPoints() == jsonTree.NumPoints());
  REQUIRE(tree.NumPoints() == binaryTree.NumPoints());

  // Check that each point is the same.
  for (size_t i = 0; i < tree.NumPoints(); ++i)
  {
    REQUIRE(tree.Point(i) == xmlTree.Point(i));
    REQUIRE(tree.Point(i) == jsonTree.Point(i));
    REQUIRE(tree.Point(i) == binaryTree.Point(i));
  }

  // Check that the parent distance is the same.
  REQUIRE(tree.ParentDistance() ==
      Approx(xmlTree.ParentDistance()).epsilon(1e-10));
  REQUIRE(tree.ParentDistance() ==
      Approx(jsonTree.ParentDistance()).epsilon(1e-10));
  REQUIRE(tree.ParentDistance() ==
      Approx(binaryTree.ParentDistance()).epsilon(1e-10));

  // Check that the furthest descendant distance is the same.
    REQUIRE(tree.FurthestDescendantDistance() ==
        Approx(xmlTree.FurthestDescendantDistance()).epsilon(1e-10));
    REQUIRE(tree.FurthestDescendantDistance() ==
        Approx(jsonTree.FurthestDescendantDistance()).epsilon(1e-10));
    REQUIRE(tree.FurthestDescendantDistance() ==
        Approx(binaryTree.FurthestDescendantDistance()).epsilon(1e-10));

  // Check that the minimum bound distance is the same.
    REQUIRE(tree.MinimumBoundDistance() ==
        Approx(xmlTree.MinimumBoundDistance()).epsilon(1e-10));
    REQUIRE(tree.MinimumBoundDistance() ==
        Approx(jsonTree.MinimumBoundDistance()).epsilon(1e-10));
    REQUIRE(tree.MinimumBoundDistance() ==
        Approx(binaryTree.MinimumBoundDistance()).epsilon(1e-10));

  // Recurse into the children.
  for (size_t i = 0; i < tree.NumChildren(); ++i)
  {
    // Check that the child dataset is the same.
    REQUIRE(&xmlTree.Dataset() == &xmlTree.Child(i).Dataset());
    REQUIRE(&jsonTree.Dataset() == &jsonTree.Child(i).Dataset());
    REQUIRE(&binaryTree.Dataset() == &binaryTree.Child(i).Dataset());

    // Make sure the parent link is right.
    REQUIRE(xmlTree.Child(i).Parent() == &xmlTree);
    REQUIRE(jsonTree.Child(i).Parent() == &jsonTree);
    REQUIRE(binaryTree.Child(i).Parent() == &binaryTree);

    CheckTrees(tree.Child(i), xmlTree.Child(i), jsonTree.Child(i),
        binaryTree.Child(i));
  }
}

TEST_CASE("BinarySpaceTreeTest", "[SerializationTest]")
{
  arma::mat data;
  data.randu(3, 100);
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  TreeType* xmlTree;
  TreeType* jsonTree;
  TreeType* binaryTree;

  SerializePointerObjectAll(&tree, xmlTree, jsonTree, binaryTree);

  CheckTrees(tree, *xmlTree, *jsonTree, *binaryTree);

  delete xmlTree;
  delete jsonTree;
  delete binaryTree;
}

TEST_CASE("BinarySpaceTreeOverwriteTest", "[SerializationTest]")
{
  arma::mat data;
  data.randu(3, 100);
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  arma::mat otherData;
  otherData.randu(5, 50);
  TreeType xmlTree(otherData);
  TreeType jsonTree(xmlTree);
  TreeType binaryTree(xmlTree);

  SerializeObjectAll(tree, xmlTree, jsonTree, binaryTree);

  CheckTrees(tree, xmlTree, jsonTree, binaryTree);
}

TEST_CASE("CoverTreeTest", "[SerializationTest]")
{
  arma::mat data;
  data.randu(3, 100);
  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType tree(data);

  TreeType* xmlTree;
  TreeType* jsonTree;
  TreeType* binaryTree;

  SerializePointerObjectAll(&tree, xmlTree, jsonTree, binaryTree);

  CheckTrees(tree, *xmlTree, *jsonTree, *binaryTree);

  // Also check a few other things.
  std::stack<TreeType*> stack, xmlStack, jsonStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(xmlTree);
  jsonStack.push(jsonTree);
  binaryStack.push(binaryTree);
  while (!stack.empty())
  {
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* jsonNode = jsonStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    jsonStack.pop();
    binaryStack.pop();

    REQUIRE(node->Scale() == xmlNode->Scale());
    REQUIRE(node->Scale() == jsonNode->Scale());
    REQUIRE(node->Scale() == binaryNode->Scale());

    REQUIRE(node->Base() == Approx(xmlNode->Base()).epsilon(1e-10));
    REQUIRE(node->Base() == Approx(jsonNode->Base()).epsilon(1e-10));
    REQUIRE(node->Base() == Approx(binaryNode->Base()).epsilon(1e-10));

    for (size_t i = 0; i < node->NumChildren(); ++i)
    {
      stack.push(&node->Child(i));
      xmlStack.push(&xmlNode->Child(i));
      jsonStack.push(&jsonNode->Child(i));
      binaryStack.push(&binaryNode->Child(i));
    }
  }

  delete xmlTree;
  delete jsonTree;
  delete binaryTree;
}

TEST_CASE("CoverTreeOverwriteTest", "[SerializationTest]")
{
  arma::mat data;
  data.randu(3, 100);
  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType tree(data);

  arma::mat otherData;
  otherData.randu(5, 50);
  TreeType xmlTree(otherData);
  TreeType jsonTree(xmlTree);
  TreeType binaryTree(xmlTree);

  SerializeObjectAll(tree, xmlTree, jsonTree, binaryTree);

  CheckTrees(tree, xmlTree, jsonTree, binaryTree);

  // Also check a few other things.
  std::stack<TreeType*> stack, xmlStack, jsonStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(&xmlTree);
  jsonStack.push(&jsonTree);
  binaryStack.push(&binaryTree);
  while (!stack.empty())
  {
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* jsonNode = jsonStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    jsonStack.pop();
    binaryStack.pop();

    REQUIRE(node->Scale() == xmlNode->Scale());
    REQUIRE(node->Scale() == jsonNode->Scale());
    REQUIRE(node->Scale() == binaryNode->Scale());

    REQUIRE(node->Base() == Approx(xmlNode->Base()).epsilon(1e-10));
    REQUIRE(node->Base() == Approx(jsonNode->Base()).epsilon(1e-10));
    REQUIRE(node->Base() == Approx(binaryNode->Base()).epsilon(1e-10));

    for (size_t i = 0; i < node->NumChildren(); ++i)
    {
      stack.push(&node->Child(i));
      xmlStack.push(&xmlNode->Child(i));
      jsonStack.push(&jsonNode->Child(i));
      binaryStack.push(&binaryNode->Child(i));
    }
  }
}

TEST_CASE("RectangleTreeTest", "[SerializationTest]")
{
  arma::mat data;
  data.randu(3, 1000);
  typedef RTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  TreeType* xmlTree;
  TreeType* jsonTree;
  TreeType* binaryTree;

  SerializePointerObjectAll(&tree, xmlTree, jsonTree, binaryTree);

  CheckTrees(tree, *xmlTree, *jsonTree, *binaryTree);

  // Check a few other things too.
  std::stack<TreeType*> stack, xmlStack, jsonStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(xmlTree);
  jsonStack.push(jsonTree);
  binaryStack.push(binaryTree);
  while (!stack.empty())
  {
    // Check more things...
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* jsonNode = jsonStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    jsonStack.pop();
    binaryStack.pop();

    REQUIRE(node->MaxLeafSize() == xmlNode->MaxLeafSize());
    REQUIRE(node->MaxLeafSize() == jsonNode->MaxLeafSize());
    REQUIRE(node->MaxLeafSize() == binaryNode->MaxLeafSize());

    REQUIRE(node->MinLeafSize() == xmlNode->MinLeafSize());
    REQUIRE(node->MinLeafSize() == jsonNode->MinLeafSize());
    REQUIRE(node->MinLeafSize() == binaryNode->MinLeafSize());

    REQUIRE(node->MaxNumChildren() == xmlNode->MaxNumChildren());
    REQUIRE(node->MaxNumChildren() == jsonNode->MaxNumChildren());
    REQUIRE(node->MaxNumChildren() == binaryNode->MaxNumChildren());

    REQUIRE(node->MinNumChildren() == xmlNode->MinNumChildren());
    REQUIRE(node->MinNumChildren() == jsonNode->MinNumChildren());
    REQUIRE(node->MinNumChildren() == binaryNode->MinNumChildren());
  }

  delete xmlTree;
  delete jsonTree;
  delete binaryTree;
}

TEST_CASE("RectangleTreeOverwriteTest", "[SerializationTest]")
{
  arma::mat data;
  data.randu(3, 1000);
  typedef RTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  arma::mat otherData;
  otherData.randu(5, 50);
  TreeType xmlTree(otherData);
  TreeType jsonTree(otherData);
  TreeType binaryTree(jsonTree);

  SerializeObjectAll(tree, xmlTree, jsonTree, binaryTree);

  CheckTrees(tree, xmlTree, jsonTree, binaryTree);

  // Check a few other things too.
  std::stack<TreeType*> stack, xmlStack, jsonStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(&xmlTree);
  jsonStack.push(&jsonTree);
  binaryStack.push(&binaryTree);
  while (!stack.empty())
  {
    // Check more things...
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* jsonNode = jsonStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    jsonStack.pop();
    binaryStack.pop();

    REQUIRE(node->MaxLeafSize() == xmlNode->MaxLeafSize());
    REQUIRE(node->MaxLeafSize() == jsonNode->MaxLeafSize());
    REQUIRE(node->MaxLeafSize() == binaryNode->MaxLeafSize());

    REQUIRE(node->MinLeafSize() == xmlNode->MinLeafSize());
    REQUIRE(node->MinLeafSize() == jsonNode->MinLeafSize());
    REQUIRE(node->MinLeafSize() == binaryNode->MinLeafSize());

    REQUIRE(node->MaxNumChildren() == xmlNode->MaxNumChildren());
    REQUIRE(node->MaxNumChildren() == jsonNode->MaxNumChildren());
    REQUIRE(node->MaxNumChildren() == binaryNode->MaxNumChildren());

    REQUIRE(node->MinNumChildren() == xmlNode->MinNumChildren());
    REQUIRE(node->MinNumChildren() == jsonNode->MinNumChildren());
    REQUIRE(node->MinNumChildren() == binaryNode->MinNumChildren());
  }
}

TEST_CASE("PerceptronTest", "[SerializationTest]")
{
  // Create a perceptron.  Train it randomly.  Then check that it hasn't
  // changed.
  arma::mat data;
  data.randu(3, 100);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < labels.n_elem; ++i)
  {
    if (data(1, i) > 0.5)
      labels[i] = 0;
    else
      labels[i] = 1;
  }

  Perceptron<> p(data, labels, 2, 15);

  Perceptron<> pXml(2, 3), pText(2, 3), pBinary(2, 3);
  SerializeObjectAll(p, pXml, pText, pBinary);

  // Now check that things are the same.
  CheckMatrices(p.Weights(), pXml.Weights(), pText.Weights(),
      pBinary.Weights());
  CheckMatrices(p.Biases(), pXml.Biases(), pText.Biases(), pBinary.Biases());

  REQUIRE(p.MaxIterations() == pXml.MaxIterations());
  REQUIRE(p.MaxIterations() == pText.MaxIterations());
  REQUIRE(p.MaxIterations() == pBinary.MaxIterations());
}

TEST_CASE("LogisticRegressionTest", "[SerializationTest]")
{
  arma::mat data;
  data.randu(3, 100);
  arma::Row<size_t> responses;
  responses.randu(100);

  LogisticRegression<> lr(data, responses, 0.5);

  LogisticRegression<> lrXml(data, responses + 3, 0.3);
  LogisticRegression<> lrText(data, responses + 1);
  LogisticRegression<> lrBinary(3, 0.0);

  SerializeObjectAll(lr, lrXml, lrText, lrBinary);

  CheckMatrices(lr.Parameters(), lrXml.Parameters(), lrText.Parameters(),
      lrBinary.Parameters());

  REQUIRE(lr.Lambda() == Approx(lrXml.Lambda()).epsilon(1e-10));
  REQUIRE(lr.Lambda() == Approx(lrText.Lambda()).epsilon(1e-10));
  REQUIRE(lr.Lambda() == Approx(lrBinary.Lambda()).epsilon(1e-10));
}

TEST_CASE("KNNTest", "[SerializationTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 2000);

  KNN knn(dataset, DUAL_TREE_MODE);

  KNN knnXml, knnText, knnBinary;

  SerializeObjectAll(knn, knnXml, knnText, knnBinary);

  // Now run nearest neighbor and make sure the results are the same.
  arma::mat querySet = arma::randu<arma::mat>(5, 1000);

  arma::mat distances, xmlDistances, jsonDistances, binaryDistances;
  arma::Mat<size_t> neighbors, xmlNeighbors, jsonNeighbors, binaryNeighbors;

  knn.Search(querySet, 5, neighbors, distances);
  knnXml.Search(querySet, 5, xmlNeighbors, xmlDistances);
  knnText.Search(querySet, 5, jsonNeighbors, jsonDistances);
  knnBinary.Search(querySet, 5, binaryNeighbors, binaryDistances);

  CheckMatrices(distances, xmlDistances, jsonDistances, binaryDistances);
  CheckMatrices(neighbors, xmlNeighbors, jsonNeighbors, binaryNeighbors);
}

TEST_CASE("SoftmaxRegressionTest", "[SerializationTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 1000);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 500; ++i)
    labels[i] = 0;
  for (size_t i = 500; i < 1000; ++i)
    labels[i] = 1;
  SoftmaxRegression sr(dataset, labels, 2);
  SoftmaxRegression srXml(dataset.n_rows, 2);
  SoftmaxRegression srText(dataset.n_rows, 2);
  SoftmaxRegression srBinary(dataset.n_rows, 2);

  SerializeObjectAll(sr, srXml, srText, srBinary);

  CheckMatrices(sr.Parameters(), srXml.Parameters(), srText.Parameters(),
      srBinary.Parameters());
}

TEST_CASE("DETTest", "[SerializationTest]")
{
  typedef DTree<arma::mat> DTreeX;

  // Create a density estimation tree on a random dataset.
  arma::mat dataset = arma::randu<arma::mat>(25, 5000);

  DTreeX tree(dataset);

  arma::mat otherDataset = arma::randu<arma::mat>(5, 100);
  DTreeX xmlTree, binaryTree, jsonTree(otherDataset);

  SerializeObjectAll(tree, xmlTree, binaryTree, jsonTree);

  std::stack<DTreeX*> stack, xmlStack, binaryStack, jsonStack;
  stack.push(&tree);
  xmlStack.push(&xmlTree);
  binaryStack.push(&binaryTree);
  jsonStack.push(&jsonTree);

  while (!stack.empty())
  {
    // Get the top node from the stack.
    DTreeX* node = stack.top();
    DTreeX* xmlNode = xmlStack.top();
    DTreeX* binaryNode = binaryStack.top();
    DTreeX* jsonNode = jsonStack.top();

    stack.pop();
    xmlStack.pop();
    binaryStack.pop();
    jsonStack.pop();

    // Check that all the members are the same.
    REQUIRE(node->Start() == xmlNode->Start());
    REQUIRE(node->Start() == binaryNode->Start());
    REQUIRE(node->Start() == jsonNode->Start());

    REQUIRE(node->End() == xmlNode->End());
    REQUIRE(node->End() == binaryNode->End());
    REQUIRE(node->End() == jsonNode->End());

    REQUIRE(node->SplitDim() == xmlNode->SplitDim());
    REQUIRE(node->SplitDim() == binaryNode->SplitDim());
    REQUIRE(node->SplitDim() == jsonNode->SplitDim());

    if (std::abs(node->SplitValue()) < 1e-5)
    {
      REQUIRE(xmlNode->SplitValue() == Approx(0.0).margin(1e-5));
      REQUIRE(binaryNode->SplitValue() == Approx(0.0).margin(1e-5));
      REQUIRE(jsonNode->SplitValue() == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(node->SplitValue() ==
          Approx(xmlNode->SplitValue()).epsilon(1e-10));
      REQUIRE(node->SplitValue() ==
          Approx(binaryNode->SplitValue()).epsilon(1e-10));
      REQUIRE(node->SplitValue() ==
          Approx(jsonNode->SplitValue()).epsilon(1e-10));
    }

    if (std::abs(node->LogNegError()) < 1e-5)
    {
      REQUIRE(xmlNode->LogNegError() == Approx(0.0).margin(1e-5));
      REQUIRE(binaryNode->LogNegError() == Approx(0.0).margin(1e-5));
      REQUIRE(jsonNode->LogNegError() == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(node->LogNegError() ==
          Approx(xmlNode->LogNegError()).epsilon(1e-10));
      REQUIRE(node->LogNegError() ==
          Approx(binaryNode->LogNegError()).epsilon(1e-10));
      REQUIRE(node->LogNegError() ==
          Approx(jsonNode->LogNegError()).epsilon(1e-10));
    }

    if (std::abs(node->SubtreeLeavesLogNegError()) < 1e-5)
    {
      REQUIRE(xmlNode->SubtreeLeavesLogNegError() == Approx(0.0).margin(1e-5));
      REQUIRE(binaryNode->SubtreeLeavesLogNegError() ==
          Approx(0.0).margin(1e-5));
      REQUIRE(jsonNode->SubtreeLeavesLogNegError() ==
          Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(node->SubtreeLeavesLogNegError() ==
          Approx(xmlNode->SubtreeLeavesLogNegError()).epsilon(1e-7));
      REQUIRE(node->SubtreeLeavesLogNegError() ==
          Approx(binaryNode->SubtreeLeavesLogNegError()).epsilon(1e-7));
      REQUIRE(node->SubtreeLeavesLogNegError() ==
          Approx(jsonNode->SubtreeLeavesLogNegError()).epsilon(1e-7));
    }

    REQUIRE(node->SubtreeLeaves() == xmlNode->SubtreeLeaves());
    REQUIRE(node->SubtreeLeaves() == binaryNode->SubtreeLeaves());
    REQUIRE(node->SubtreeLeaves() == jsonNode->SubtreeLeaves());

    if (std::abs(node->Ratio()) < 1e-5)
    {
      REQUIRE(xmlNode->Ratio() == Approx(0.0).margin(1e-5));
      REQUIRE(binaryNode->Ratio() == Approx(0.0).margin(1e-5));
      REQUIRE(jsonNode->Ratio() == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(node->Ratio() == Approx(xmlNode->Ratio()).epsilon(1e-10));
      REQUIRE(node->Ratio() ==
          Approx(binaryNode->Ratio()).epsilon(1e-10));
      REQUIRE(node->Ratio() == Approx(jsonNode->Ratio()).epsilon(1e-10));
    }

    if (std::abs(node->LogVolume()) < 1e-5)
    {
      REQUIRE(xmlNode->LogVolume() == Approx(0.0).margin(1e-5));
      REQUIRE(binaryNode->LogVolume() == Approx(0.0).margin(1e-5));
      REQUIRE(jsonNode->LogVolume() == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(node->LogVolume() == Approx(xmlNode->LogVolume()).epsilon(1e-10));
      REQUIRE(node->LogVolume() ==
          Approx(binaryNode->LogVolume()).epsilon(1e-10));
      REQUIRE(node->LogVolume() ==
          Approx(jsonNode->LogVolume()).epsilon(1e-10));
    }

    if (node->Left() == NULL)
    {
      REQUIRE(xmlNode->Left() == NULL);
      REQUIRE(binaryNode->Left() == NULL);
      REQUIRE(jsonNode->Left() == NULL);
    }
    else
    {
      REQUIRE(xmlNode->Left() != NULL);
      REQUIRE(binaryNode->Left() != NULL);
      REQUIRE(jsonNode->Left() != NULL);

      // Push children onto stack.
      stack.push(node->Left());
      xmlStack.push(xmlNode->Left());
      binaryStack.push(binaryNode->Left());
      jsonStack.push(jsonNode->Left());
    }

    if (node->Right() == NULL)
    {
      REQUIRE(xmlNode->Right() == NULL);
      REQUIRE(binaryNode->Right() == NULL);
      REQUIRE(jsonNode->Right() == NULL);
    }
    else
    {
      REQUIRE(xmlNode->Right() != NULL);
      REQUIRE(binaryNode->Right() != NULL);
      REQUIRE(jsonNode->Right() != NULL);

      // Push children onto stack.
      stack.push(node->Right());
      xmlStack.push(xmlNode->Right());
      binaryStack.push(binaryNode->Right());
      jsonStack.push(jsonNode->Right());
    }

    REQUIRE(node->Root() == xmlNode->Root());
    REQUIRE(node->Root() == binaryNode->Root());
    REQUIRE(node->Root() == jsonNode->Root());

    if (std::abs(node->AlphaUpper()) < 1e-5)
    {
      REQUIRE(xmlNode->AlphaUpper() == Approx(0.0).margin(1e-5));
      REQUIRE(binaryNode->AlphaUpper() == Approx(0.0).margin(1e-5));
      REQUIRE(jsonNode->AlphaUpper() == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(node->AlphaUpper() ==
          Approx(xmlNode->AlphaUpper()).epsilon(1e-10));
      REQUIRE(node->AlphaUpper() ==
          Approx(binaryNode->AlphaUpper()).epsilon(1e-10));
      REQUIRE(node->AlphaUpper() ==
          Approx(jsonNode->AlphaUpper()).epsilon(1e-10));
    }

    REQUIRE(node->MaxVals().n_elem == xmlNode->MaxVals().n_elem);
    REQUIRE(node->MaxVals().n_elem == binaryNode->MaxVals().n_elem);
    REQUIRE(node->MaxVals().n_elem == jsonNode->MaxVals().n_elem);
    for (size_t i = 0; i < node->MaxVals().n_elem; ++i)
    {
      if (std::abs(node->MaxVals()[i]) < 1e-5)
      {
        REQUIRE(xmlNode->MaxVals()[i] == Approx(0.0).margin(1e-5));
        REQUIRE(binaryNode->MaxVals()[i] == Approx(0.0).margin(1e-5));
        REQUIRE(jsonNode->MaxVals()[i] == Approx(0.0).margin(1e-5));
      }
      else
      {
        REQUIRE(node->MaxVals()[i] ==
            Approx(xmlNode->MaxVals()[i]).epsilon(1e-10));
        REQUIRE(node->MaxVals()[i] ==
            Approx(binaryNode->MaxVals()[i]).epsilon(1e-10));
        REQUIRE(node->MaxVals()[i] ==
            Approx(jsonNode->MaxVals()[i]).epsilon(1e-10));
      }
    }

    REQUIRE(node->MinVals().n_elem == xmlNode->MinVals().n_elem);
    REQUIRE(node->MinVals().n_elem == binaryNode->MinVals().n_elem);
    REQUIRE(node->MinVals().n_elem == jsonNode->MinVals().n_elem);
    for (size_t i = 0; i < node->MinVals().n_elem; ++i)
    {
      if (std::abs(node->MinVals()[i]) < 1e-5)
      {
        REQUIRE(xmlNode->MinVals()[i] == Approx(0.0).margin(1e-5));
        REQUIRE(binaryNode->MinVals()[i] == Approx(0.0).margin(1e-5));
        REQUIRE(jsonNode->MinVals()[i] == Approx(0.0).margin(1e-5));
      }
      else
      {
        REQUIRE(node->MinVals()[i] ==
            Approx(xmlNode->MinVals()[i]).epsilon(1e-10));
        REQUIRE(node->MinVals()[i] ==
            Approx(binaryNode->MinVals()[i]).epsilon(1e-10));
        REQUIRE(node->MinVals()[i] ==
            Approx(jsonNode->MinVals()[i]).epsilon(1e-10));
      }
    }
  }
}

TEST_CASE("NaiveBayesSerializationTest", "[SerializationTest]")
{
  // Train NBC randomly.  Make sure the model is the same after serializing and
  // re-loading.
  arma::mat dataset;
  dataset.randu(10, 500);
  arma::Row<size_t> labels(500);
  for (size_t i = 0; i < 500; ++i)
  {
    if (dataset(0, i) > 0.5)
      labels[i] = 0;
    else
      labels[i] = 1;
  }

  NaiveBayesClassifier<> nbc(dataset, labels, 2);

  // Initialize some empty Naive Bayes classifiers.
  NaiveBayesClassifier<> xmlNbc(0, 0), jsonNbc(0, 0), binaryNbc(0, 0);
  SerializeObjectAll(nbc, xmlNbc, jsonNbc, binaryNbc);

  REQUIRE(nbc.Means().n_elem == xmlNbc.Means().n_elem);
  REQUIRE(nbc.Means().n_elem == jsonNbc.Means().n_elem);
  REQUIRE(nbc.Means().n_elem == binaryNbc.Means().n_elem);
  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    REQUIRE(nbc.Means()[i] == Approx(xmlNbc.Means()[i]).epsilon(1e-10));
    REQUIRE(nbc.Means()[i] == Approx(jsonNbc.Means()[i]).epsilon(1e-10));
    REQUIRE(nbc.Means()[i] == Approx(binaryNbc.Means()[i]).epsilon(1e-10));
  }
  REQUIRE(nbc.Variances().n_elem == xmlNbc.Variances().n_elem);
  REQUIRE(nbc.Variances().n_elem == jsonNbc.Variances().n_elem);
  REQUIRE(nbc.Variances().n_elem == binaryNbc.Variances().n_elem);
  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    REQUIRE(nbc.Variances()[i] ==
        Approx(xmlNbc.Variances()[i]).epsilon(1e-10));
    REQUIRE(nbc.Variances()[i] ==
        Approx(jsonNbc.Variances()[i]).epsilon(1e-10));
    REQUIRE(nbc.Variances()[i] ==
        Approx(binaryNbc.Variances()[i]).epsilon(1e-10));
  }

  REQUIRE(nbc.Probabilities().n_elem ==
      xmlNbc.Probabilities().n_elem);
  REQUIRE(nbc.Probabilities().n_elem ==
      jsonNbc.Probabilities().n_elem);
  REQUIRE(nbc.Probabilities().n_elem ==
      binaryNbc.Probabilities().n_elem);
  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    REQUIRE(nbc.Probabilities()[i] ==
        Approx(xmlNbc.Probabilities()[i]).epsilon(1e-7));
    REQUIRE(nbc.Probabilities()[i] ==
        Approx(jsonNbc.Probabilities()[i]).epsilon(1e-7));
    REQUIRE(nbc.Probabilities()[i] ==
        Approx(binaryNbc.Probabilities()[i]).epsilon(1e-7));
  }
}

TEST_CASE("RASearchTest", "[SerializationTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 200);
  arma::mat otherDataset = arma::randu<arma::mat>(5, 100);

  // Find nearest neighbors in the top 10, with accuracy 0.95.  So 95% of the
  // results we get (at least) should fall into the top 10 of the true nearest
  // neighbors.
  KRANN allkrann(dataset, false, false, 5, 0.95);

  KRANN krannXml(otherDataset, false, false);
  KRANN krannText(otherDataset, true, false);
  KRANN krannBinary(otherDataset, true, true);

  SerializeObjectAll(allkrann, krannXml, krannText, krannBinary);

  // Now run nearest neighbor and make sure the results are the same.
  arma::mat querySet = arma::randu<arma::mat>(5, 100);

  arma::mat distances, xmlDistances, jsonDistances, binaryDistances;
  arma::Mat<size_t> neighbors, xmlNeighbors, jsonNeighbors, binaryNeighbors;

  KNN knn(dataset); // Exact search.
  knn.Search(querySet, 10, neighbors, distances);
  krannXml.Search(querySet, 5, xmlNeighbors, xmlDistances);
  krannText.Search(querySet, 5, jsonNeighbors, jsonDistances);
  krannBinary.Search(querySet, 5, binaryNeighbors, binaryDistances);

  REQUIRE(xmlNeighbors.n_rows == 5);
  REQUIRE(xmlNeighbors.n_cols == 100);
  REQUIRE(jsonNeighbors.n_rows == 5);
  REQUIRE(jsonNeighbors.n_cols == 100);
  REQUIRE(binaryNeighbors.n_rows == 5);
  REQUIRE(binaryNeighbors.n_cols == 100);

  size_t xmlCorrect = 0;
  size_t jsonCorrect = 0;
  size_t binaryCorrect = 0;
  for (size_t i = 0; i < xmlNeighbors.n_cols; ++i)
  {
    // See how many are in the top 10.
    for (size_t j = 0; j < xmlNeighbors.n_rows; ++j)
    {
      for (size_t k = 0; k < neighbors.n_rows; ++k)
      {
        if (neighbors(k, i) == xmlNeighbors(j, i))
          xmlCorrect++;
        if (neighbors(k, i) == jsonNeighbors(j, i))
          jsonCorrect++;
        if (neighbors(k, i) == binaryNeighbors(j, i))
          binaryCorrect++;
      }
    }
  }

  // We need 95% of these to be correct.
  REQUIRE(xmlCorrect > (95 * 5));
  REQUIRE(binaryCorrect > (95 * 5));
  REQUIRE(jsonCorrect > (95 * 5));
}

/**
 * Test that an LSH model can be serialized and deserialized.
 */
TEST_CASE("LSHTest", "[SerializationTest]")
{
  // Since we still don't have good tests for LSH, basically what we're going to
  // do is serialize an LSH model, and make sure we can deserialize it and that
  // we still get results when we call Search().
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);

  LSHSearch<> lsh(referenceData, 5, 10); // Arbitrary chosen parameters.

  LSHSearch<> xmlLsh;
  arma::mat jsonData = arma::randu<arma::mat>(5, 50);
  LSHSearch<> jsonLsh(jsonData, 4, 5);
  LSHSearch<> binaryLsh(referenceData, 15, 2);

  // Now serialize.
  SerializeObjectAll(lsh, xmlLsh, jsonLsh, binaryLsh);

  // Check what we can about the serialized objects.
  REQUIRE(lsh.NumProjections() == xmlLsh.NumProjections());
  REQUIRE(lsh.NumProjections() == jsonLsh.NumProjections());
  REQUIRE(lsh.NumProjections() == binaryLsh.NumProjections());
  for (size_t i = 0; i < lsh.NumProjections(); ++i)
  {
    CheckMatrices(lsh.Projections().slice(i), xmlLsh.Projections().slice(i),
        jsonLsh.Projections().slice(i), binaryLsh.Projections().slice(i));
  }

  CheckMatrices(lsh.ReferenceSet(), xmlLsh.ReferenceSet(),
      jsonLsh.ReferenceSet(), binaryLsh.ReferenceSet());
  CheckMatrices(lsh.Offsets(), xmlLsh.Offsets(), jsonLsh.Offsets(),
      binaryLsh.Offsets());
  CheckMatrices(lsh.SecondHashWeights(), xmlLsh.SecondHashWeights(),
      jsonLsh.SecondHashWeights(), binaryLsh.SecondHashWeights());

  REQUIRE(lsh.BucketSize() == xmlLsh.BucketSize());
  REQUIRE(lsh.BucketSize() == jsonLsh.BucketSize());
  REQUIRE(lsh.BucketSize() == binaryLsh.BucketSize());

  REQUIRE(lsh.SecondHashTable().size() ==
      xmlLsh.SecondHashTable().size());
  REQUIRE(lsh.SecondHashTable().size() ==
      jsonLsh.SecondHashTable().size());
  REQUIRE(lsh.SecondHashTable().size() ==
      binaryLsh.SecondHashTable().size());

  for (size_t i = 0; i < lsh.SecondHashTable().size(); ++i)
  CheckMatrices(lsh.SecondHashTable()[i], xmlLsh.SecondHashTable()[i],
      jsonLsh.SecondHashTable()[i], binaryLsh.SecondHashTable()[i]);
}

// Make sure serialization works for LARS.
TEST_CASE("LARSTest", "[SerializationTest]")
{
  // Create a dataset.
  arma::mat X = arma::randn(75, 250);
  arma::vec beta = arma::randn(75, 1);
  arma::rowvec y = beta.t() * X;

  LARS lars(true, 0.1, 0.1);
  arma::vec betaOpt;
  lars.Train(X, y, betaOpt);

  // Now, serialize.
  LARS xmlLars(false, 0.5, 0.0), binaryLars(true, 1.0, 0.0),
      jsonLars(false, 0.1, 0.1);

  // Train jsonLars.
  arma::mat jsonX = arma::randn(25, 150);
  arma::vec jsonBeta = arma::randn(25, 1);
  arma::rowvec jsonY = jsonBeta.t() * jsonX;
  arma::vec jsonBetaOpt;
  jsonLars.Train(jsonX, jsonY, jsonBetaOpt);

  SerializeObjectAll(lars, xmlLars, binaryLars, jsonLars);

  // Now, check that predictions are the same.
  arma::rowvec pred, xmlPred, jsonPred, binaryPred;
  lars.Predict(X, pred);
  xmlLars.Predict(X, xmlPred);
  jsonLars.Predict(X, jsonPred);
  binaryLars.Predict(X, binaryPred);

  CheckMatrices(pred, xmlPred, jsonPred, binaryPred);
}

/**
 * Test serialization of the HoeffdingNumericSplit object after binning has
 * occured.
 */
TEST_CASE("HoeffdingNumericSplitTest", "[SerializationTest]")
{
  HoeffdingNumericSplit<GiniImpurity> split(3);
  // Train until it bins.
  for (size_t i = 0; i < 200; ++i)
    split.Train(Random(), RandInt(3));

  HoeffdingNumericSplit<GiniImpurity> xmlSplit(5);
  HoeffdingNumericSplit<GiniImpurity> jsonSplit(7);
  for (size_t i = 0; i < 200; ++i)
    jsonSplit.Train(Random() + 3, 0);
  HoeffdingNumericSplit<GiniImpurity> binarySplit(2);

  SerializeObjectAll(split, xmlSplit, jsonSplit, binarySplit);

  // Ensure that everything is the same.
  REQUIRE(split.Bins() == xmlSplit.Bins());
  REQUIRE(split.Bins() == jsonSplit.Bins());
  REQUIRE(split.Bins() == binarySplit.Bins());

  double bestSplit, secondBestSplit;
  double baseBestSplit, baseSecondBestSplit;
  split.EvaluateFitnessFunction(baseBestSplit, baseSecondBestSplit);
  xmlSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  REQUIRE(bestSplit == Approx(baseBestSplit).epsilon(1e-10));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-10));

  jsonSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  REQUIRE(bestSplit == Approx(baseBestSplit).epsilon(1e-10));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-10));

  binarySplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  REQUIRE(bestSplit == Approx(baseBestSplit).epsilon(1e-10));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-10));

  arma::Col<size_t> children, xmlChildren, jsonChildren, binaryChildren;
  NumericSplitInfo<double> splitInfo, xmlSplitInfo, jsonSplitInfo,
      binarySplitInfo;

  split.Split(children, splitInfo);
  xmlSplit.Split(xmlChildren, xmlSplitInfo);
  binarySplit.Split(binaryChildren, binarySplitInfo);
  jsonSplit.Split(jsonChildren, jsonSplitInfo);

  REQUIRE(children.size() == xmlChildren.size());
  REQUIRE(children.size() == jsonChildren.size());
  REQUIRE(children.size() == binaryChildren.size());
  for (size_t i = 0; i < children.size(); ++i)
  {
    REQUIRE(children[i] == xmlChildren[i]);
    REQUIRE(children[i] == jsonChildren[i]);
    REQUIRE(children[i] == binaryChildren[i]);
  }

  // Random checks.
  for (size_t i = 0; i < 200; ++i)
  {
    const double random = Random() * 1.5;
    REQUIRE(splitInfo.CalculateDirection(random) ==
                        xmlSplitInfo.CalculateDirection(random));
    REQUIRE(splitInfo.CalculateDirection(random) ==
                        jsonSplitInfo.CalculateDirection(random));
    REQUIRE(splitInfo.CalculateDirection(random) ==
                        binarySplitInfo.CalculateDirection(random));
  }
}

/**
 * Make sure serialization of the HoeffdingNumericSplit object before binning
 * occurs is successful.
 */
TEST_CASE("HoeffdingNumericSplitBeforeBinningTest", "[SerializationTest]")
{
  HoeffdingNumericSplit<GiniImpurity> split(3);
  // Train but not until it bins.
  for (size_t i = 0; i < 50; ++i)
    split.Train(Random(), RandInt(3));

  HoeffdingNumericSplit<GiniImpurity> xmlSplit(5);
  HoeffdingNumericSplit<GiniImpurity> jsonSplit(7);
  for (size_t i = 0; i < 200; ++i)
    jsonSplit.Train(Random() + 3, 0);
  HoeffdingNumericSplit<GiniImpurity> binarySplit(2);

  SerializeObjectAll(split, xmlSplit, jsonSplit, binarySplit);

  // Ensure that everything is the same.
  REQUIRE(split.Bins() == xmlSplit.Bins());
  REQUIRE(split.Bins() == jsonSplit.Bins());
  REQUIRE(split.Bins() == binarySplit.Bins());

  double baseBestSplit, baseSecondBestSplit;
  double bestSplit, secondBestSplit;
  split.EvaluateFitnessFunction(baseBestSplit, baseSecondBestSplit);
  jsonSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);

  REQUIRE(baseBestSplit == Approx(0.0).margin(1e-5));
  REQUIRE(baseSecondBestSplit == Approx(0.0).margin(1e-5));

  REQUIRE(bestSplit == Approx(0.0).margin(1e-5));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-5));

  xmlSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  REQUIRE(bestSplit == Approx(0.0).margin(1e-5));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-5));

  binarySplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  REQUIRE(bestSplit == Approx(0.0).margin(1e-5));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-5));
}

/**
 * Make sure the HoeffdingCategoricalSplit object serializes correctly.
 */
TEST_CASE("HoeffdingCategoricalSplitTest", "[SerializationTest]")
{
  HoeffdingCategoricalSplit<GiniImpurity> split(10, 3);
  for (size_t i = 0; i < 50; ++i)
    split.Train(RandInt(10), RandInt(3));

  HoeffdingCategoricalSplit<GiniImpurity> xmlSplit(3, 7);
  HoeffdingCategoricalSplit<GiniImpurity> binarySplit(4, 11);
  HoeffdingCategoricalSplit<GiniImpurity> jsonSplit(2, 2);
  for (size_t i = 0; i < 10; ++i)
    jsonSplit.Train(RandInt(2), RandInt(2));

  SerializeObjectAll(split, xmlSplit, jsonSplit, binarySplit);

  REQUIRE(split.MajorityClass() == xmlSplit.MajorityClass());
  REQUIRE(split.MajorityClass() == jsonSplit.MajorityClass());
  REQUIRE(split.MajorityClass() == binarySplit.MajorityClass());

  double bestSplit, secondBestSplit;
  double baseBestSplit, baseSecondBestSplit;
  split.EvaluateFitnessFunction(baseBestSplit, baseSecondBestSplit);
  xmlSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);

  REQUIRE(bestSplit == Approx(baseBestSplit).epsilon(1e-10));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-10));

  jsonSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  REQUIRE(bestSplit == Approx(baseBestSplit).epsilon(1e-10));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-10));

  binarySplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  REQUIRE(bestSplit == Approx(baseBestSplit).epsilon(1e-10));
  REQUIRE(secondBestSplit == Approx(0.0).margin(1e-10));

  arma::Col<size_t> children, xmlChildren, jsonChildren, binaryChildren;
  CategoricalSplitInfo splitInfo(1); // I don't care about this.

  split.Split(children, splitInfo);
  xmlSplit.Split(xmlChildren, splitInfo);
  binarySplit.Split(binaryChildren, splitInfo);
  jsonSplit.Split(jsonChildren, splitInfo);

  REQUIRE(children.size() == xmlChildren.size());
  REQUIRE(children.size() == jsonChildren.size());
  REQUIRE(children.size() == binaryChildren.size());
  for (size_t i = 0; i < children.size(); ++i)
  {
    REQUIRE(children[i] == xmlChildren[i]);
    REQUIRE(children[i] == jsonChildren[i]);
    REQUIRE(children[i] == binaryChildren[i]);
  }
}

/**
 * Make sure the HoeffdingTree object serializes correctly before a split has
 * occured.
 */
TEST_CASE("HoeffdingTreeBeforeSplitTest", "[SerializationTest]")
{
  data::DatasetInfo info(5);
  info.MapString<double>("0", 2); // Dimension 1 is categorical.
  info.MapString<double>("1", 2);
  HoeffdingTree<> split(info, 2, 0.99, 15000, 1);

  // Train for 2 samples.
  split.Train(arma::vec("0.3 0.4 1 0.6 0.7"), 0);
  split.Train(arma::vec("-0.3 0.0 0 0.7 0.8"), 1);

  data::DatasetInfo wrongInfo(3);
  wrongInfo.MapString<double>("1", 1);
  HoeffdingTree<> xmlSplit(wrongInfo, 7, 0.1, 10, 1);

  // Force the binarySplit to split.
  data::DatasetInfo binaryInfo(2);
  binaryInfo.MapString<double>("cat0", 0);
  binaryInfo.MapString<double>("cat1", 0);
  binaryInfo.MapString<double>("cat0", 1);

  HoeffdingTree<> binarySplit(info, 2, 0.95, 5000, 1);

  // Feed samples from each class.
  for (size_t i = 0; i < 500; ++i)
  {
    binarySplit.Train(arma::Col<size_t>("0 0"), 0);
    binarySplit.Train(arma::Col<size_t>("1 0"), 1);
  }

  HoeffdingTree<> jsonSplit(wrongInfo, 11, 0.75, 1000, 1);

  SerializeObjectAll(split, xmlSplit, jsonSplit, binarySplit);

  REQUIRE(split.SplitDimension() == xmlSplit.SplitDimension());
  REQUIRE(split.SplitDimension() == binarySplit.SplitDimension());
  REQUIRE(split.SplitDimension() == jsonSplit.SplitDimension());

  REQUIRE(split.MajorityClass() == xmlSplit.MajorityClass());
  REQUIRE(split.MajorityClass() == binarySplit.MajorityClass());
  REQUIRE(split.MajorityClass() == jsonSplit.MajorityClass());

  REQUIRE(split.SplitCheck() == xmlSplit.SplitCheck());
  REQUIRE(split.SplitCheck() == binarySplit.SplitCheck());
  REQUIRE(split.SplitCheck() == jsonSplit.SplitCheck());
}

/**
 * Make sure the HoeffdingTree object serializes correctly after a split has
 * occurred.
 */
TEST_CASE("HoeffdingTreeAfterSplitTest", "[SerializationTest]")
{
  // Force the split to split.
  data::DatasetInfo info(2);
  info.MapString<double>("cat0", 0);
  info.MapString<double>("cat1", 0);
  info.MapString<double>("cat0", 1);

  HoeffdingTree<> split(info, 2, 0.95, 5000, 1);

  // Feed samples from each class.
  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(arma::Col<size_t>("0 0"), 0);
    split.Train(arma::Col<size_t>("1 0"), 1);
  }
  // Ensure a split has happened.
  REQUIRE(split.SplitDimension() != size_t(-1));

  data::DatasetInfo wrongInfo(3);
  wrongInfo.MapString<double>("1", 1);
  HoeffdingTree<> xmlSplit(wrongInfo, 7, 0.1, 10, 1);

  data::DatasetInfo binaryInfo(5);
  binaryInfo.MapString<double>("0", 2); // Dimension 2 is categorical.
  binaryInfo.MapString<double>("1", 2);
  HoeffdingTree<> binarySplit(binaryInfo, 2, 0.99, 15000, 1);

  // Train for 2 samples.
  binarySplit.Train(arma::vec("0.3 0.4 1 0.6 0.7"), 0);
  binarySplit.Train(arma::vec("-0.3 0.0 0 0.7 0.8"), 1);

  HoeffdingTree<> jsonSplit(wrongInfo, 11, 0.75, 1000, 1);

  SerializeObjectAll(split, xmlSplit, jsonSplit, binarySplit);

  REQUIRE(split.SplitDimension() == xmlSplit.SplitDimension());
  REQUIRE(split.SplitDimension() == binarySplit.SplitDimension());
  REQUIRE(split.SplitDimension() == jsonSplit.SplitDimension());

  // If splitting has already happened, then SplitCheck() should return 0.
  REQUIRE(split.SplitCheck() == 0);
  REQUIRE(split.SplitCheck() == xmlSplit.SplitCheck());
  REQUIRE(split.SplitCheck() == binarySplit.SplitCheck());
  REQUIRE(split.SplitCheck() == jsonSplit.SplitCheck());

  REQUIRE(split.MajorityClass() == xmlSplit.MajorityClass());
  REQUIRE(split.MajorityClass() == binarySplit.MajorityClass());
  REQUIRE(split.MajorityClass() == jsonSplit.MajorityClass());

  REQUIRE(split.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")) ==
      xmlSplit.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")));
  REQUIRE(split.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")) ==
      binarySplit.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")));
  REQUIRE(split.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")) ==
      jsonSplit.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")));
}

TEST_CASE("EmptyHoeffdingTreeTest", "[SerializationTest]")
{
  data::DatasetInfo info(6);
  HoeffdingTree<> tree(info, 2);
  HoeffdingTree<> xmlTree(info, 3);
  HoeffdingTree<> binaryTree(info, 4);
  HoeffdingTree<> jsonTree(info, 5);

  SerializeObjectAll(tree, xmlTree, binaryTree, jsonTree);

  REQUIRE(tree.NumChildren() == 0);
  REQUIRE(xmlTree.NumChildren() == 0);
  REQUIRE(binaryTree.NumChildren() == 0);
  REQUIRE(jsonTree.NumChildren() == 0);
}

/**
 * Build a Hoeffding tree, then save it and make sure other trees can classify
 * as effectively.
 */
TEST_CASE("HoeffdingTreeTest", "[SerializationTest]")
{
  arma::mat dataset(2, 400);
  arma::Row<size_t> labels(400);
  for (size_t i = 0; i < 200; ++i)
  {
    dataset(0, 2 * i) = RandInt(4);
    dataset(1, 2 * i) = RandInt(2);
    dataset(0, 2 * i + 1) = RandInt(4);
    dataset(1, 2 * i + 1) = RandInt(2) + 2;
    labels[2 * i] = 0;
    labels[2 * i + 1] = 1;
  }
  // Make the features categorical.
  data::DatasetInfo info(2);
  info.MapString<double>("a", 0);
  info.MapString<double>("b", 0);
  info.MapString<double>("c", 0);
  info.MapString<double>("d", 0);
  info.MapString<double>("a", 1);
  info.MapString<double>("b", 1);
  info.MapString<double>("c", 1);
  info.MapString<double>("d", 1);

  HoeffdingTree<> tree(dataset, info, labels, 2, false /* no batch mode */);

  data::DatasetInfo xmlInfo(1);
  HoeffdingTree<> xmlTree(xmlInfo, 1);
  data::DatasetInfo binaryInfo(5);
  HoeffdingTree<> binaryTree(binaryInfo, 6);
  data::DatasetInfo jsonInfo(7);
  HoeffdingTree<> jsonTree(jsonInfo, 100);

  SerializeObjectAll(tree, xmlTree, jsonTree, binaryTree);

  REQUIRE(tree.NumChildren() == xmlTree.NumChildren());
  REQUIRE(tree.NumChildren() == jsonTree.NumChildren());
  REQUIRE(tree.NumChildren() == binaryTree.NumChildren());

  REQUIRE(tree.SplitDimension() == xmlTree.SplitDimension());
  REQUIRE(tree.SplitDimension() == jsonTree.SplitDimension());
  REQUIRE(tree.SplitDimension() == binaryTree.SplitDimension());

  for (size_t i = 0; i < tree.NumChildren(); ++i)
  {
    REQUIRE(tree.Child(i).NumChildren() == 0);
    REQUIRE(xmlTree.Child(i).NumChildren() == 0);
    REQUIRE(binaryTree.Child(i).NumChildren() == 0);
    REQUIRE(jsonTree.Child(i).NumChildren() == 0);

    REQUIRE(tree.Child(i).SplitDimension() ==
        xmlTree.Child(i).SplitDimension());
    REQUIRE(tree.Child(i).SplitDimension() ==
        jsonTree.Child(i).SplitDimension());
    REQUIRE(tree.Child(i).SplitDimension() ==
        binaryTree.Child(i).SplitDimension());
  }
}

/**
 * Build a Binary RBM, then save it and make sure the parameters of the
 * all the RBM are equal.
 *
TEST_CASE("BinaryRBMTest", "[SerializationTest]")
{
  arma::mat data;
  size_t hiddenLayerSize = 5;
  data.randu(3, 100);

  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization> Rbm(data, gaussian, data.n_rows, hiddenLayerSize,
      1, 1, 1, 2, 8, 1, true);
  RBM<GaussianInitialization> RbmXml(data, gaussian, data.n_rows,
      hiddenLayerSize, 1, 1, 1, 2, 8, 1, true);
  RBM<GaussianInitialization> RbmText(data, gaussian, data.n_rows,
      hiddenLayerSize, 1, 1, 1, 2, 8, 1, true);
  RBM<GaussianInitialization> RbmBinary(data, gaussian, data.n_rows,
      hiddenLayerSize, 1, 1, 1, 2, 8, 1, true);
  Rbm.Reset();

  SerializeObjectAll(Rbm, RbmXml, RbmText, RbmBinary);
  CheckMatrices(Rbm.Parameters(), RbmXml.Parameters(), RbmText.Parameters(),
      RbmBinary.Parameters());
  CheckMatrices(Rbm.VisibleBias(), RbmXml.VisibleBias());
  CheckMatrices(Rbm.VisibleBias(), RbmText.VisibleBias());
  CheckMatrices(Rbm.VisibleBias(), RbmBinary.VisibleBias());

  CheckMatrices(Rbm.HiddenBias(), RbmXml.HiddenBias());
  CheckMatrices(Rbm.HiddenBias(), RbmText.HiddenBias());
  CheckMatrices(Rbm.HiddenBias(), RbmBinary.HiddenBias());

  CheckMatrices(Rbm.Weight(), RbmXml.Weight());
  CheckMatrices(Rbm.Weight(), RbmText.Weight());
  CheckMatrices(Rbm.Weight(), RbmBinary.Weight());
}
*/

/**
 * Build a ssRBM, then save it and make sure the parameters of the
 * all the RBM are equal.
 *
TEST_CASE("ssRBMTest", "[SerializationTest]")
{
  arma::mat data;
  size_t hiddenLayerSize = 5;
  data.randu(3, 100);
  double slabPenalty = 1;
  double tempRadius, radius = arma::norm(data.col(0));
  for (size_t i = 1; i < data.n_cols; ++i)
  {
    tempRadius = arma::norm(data.col(i));
    if (radius < tempRadius)
      radius = tempRadius;
  }

  size_t poolSize = 1;

  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, arma::mat, SpikeSlabRBM> Rbm(data, gaussian,
      data.n_rows, hiddenLayerSize, 1, 1, 1, poolSize, slabPenalty, radius,
      true);
  RBM<GaussianInitialization, arma::mat, SpikeSlabRBM> RbmXml(data, gaussian,
      data.n_rows, hiddenLayerSize, 1, 1, 1, poolSize, slabPenalty, radius,
      true);
  RBM<GaussianInitialization, arma::mat, SpikeSlabRBM> RbmText(data, gaussian,
      data.n_rows, hiddenLayerSize, 1, 1, 1, poolSize, slabPenalty, radius,
      true);
  RBM<GaussianInitialization, arma::mat, SpikeSlabRBM> RbmBinary(data, gaussian,
      data.n_rows, hiddenLayerSize, 1, 1, 1, poolSize, slabPenalty, radius,
      true);
  Rbm.Reset();
  Rbm.VisiblePenalty().fill(15);
  Rbm.SpikeBias().ones();

  SerializeObjectAll(Rbm, RbmXml, RbmText, RbmBinary);
  CheckMatrices(Rbm.Parameters(), RbmXml.Parameters(), RbmText.Parameters(),
      RbmBinary.Parameters());

  CheckMatrices(Rbm.VisiblePenalty(), RbmXml.VisiblePenalty());
  CheckMatrices(Rbm.VisiblePenalty(), RbmText.VisiblePenalty());
  CheckMatrices(Rbm.VisiblePenalty(), RbmBinary.VisiblePenalty());

  CheckMatrices(Rbm.SpikeBias(), RbmXml.SpikeBias());
  CheckMatrices(Rbm.SpikeBias(), RbmText.SpikeBias());
  CheckMatrices(Rbm.SpikeBias(), RbmBinary.SpikeBias());

  CheckMatrices(Rbm.Weight(), RbmXml.Weight());
  CheckMatrices(Rbm.Weight(), RbmText.Weight());
  CheckMatrices(Rbm.Weight(), RbmBinary.Weight());
}
*/

// Make sure serialization works for BayesianLinearRegression.
TEST_CASE("BayesianLinearRegressionTest", "[SerializationTest]")
{
  // Create a dataset.
  arma::mat matX = arma::randn(75, 250);
  arma::vec omega = arma::randn(75, 1);
  arma::rowvec y = omega.t() * matX;

  BayesianLinearRegression blr(false, false);
  blr.Train(matX, y);
  arma::vec omegaOpt = blr.Omega();

  // Now, serialize.
  BayesianLinearRegression xmlBlr(false, false), binaryBlr(false, false),
    textBlr(false, false);

  SerializeObjectAll(blr, xmlBlr, binaryBlr, textBlr);

  // Now, check that predictions are the same.
  arma::rowvec pred, xmlPred, textPred, binaryPred;
  blr.Predict(matX, pred);
  xmlBlr.Predict(matX, xmlPred);
  textBlr.Predict(matX, textPred);
  binaryBlr.Predict(matX, binaryPred);

  CheckMatrices(pred, xmlPred, textPred, binaryPred);
}

/**
 * Test the cereal array wrapper on an empty array.
 */
class TestStruct
{
 public:
  TestStruct() : mem(NULL), len(0) { }

  template<typename Archive>
  void serialize(Archive& ar)
  {
    ar(cereal::make_array(mem, len));
  }

  int* mem;
  size_t len;
};

TEST_CASE("CerealEmptyArrayWrapperTest", "[SerializationTest]")
{
  TestStruct t;
  // Manually change the values in the other ones.
  TestStruct xmlT, jsonT, binaryT;
  xmlT.mem = new int[10];
  xmlT.len = 10;
  jsonT.mem = new int[5];
  jsonT.len = 5;

  SerializeObjectAll(t, xmlT, jsonT, binaryT);

  // Ensure that all the results are correct.
  REQUIRE(xmlT.mem == (int*) NULL);
  REQUIRE(xmlT.len == 0);
  REQUIRE(binaryT.mem == (int*) NULL);
  REQUIRE(binaryT.len == 0);
  REQUIRE(jsonT.mem == (int*) NULL);
  REQUIRE(jsonT.len == 0);
}
