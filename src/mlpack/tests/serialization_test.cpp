/**
 * @file serialization_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

#include <mlpack/core/dists/regression_distribution.hpp>
#include <mlpack/core/tree/ballbound.hpp>
#include <mlpack/core/tree/hrectbound.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>

#include <mlpack/methods/perceptron/perceptron.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/det/dtree.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/rann/ra_search.hpp>
#include <mlpack/methods/lsh/lsh_search.hpp>
#include <mlpack/methods/decision_stump/decision_stump.hpp>
#include <mlpack/methods/lars/lars.hpp>

using namespace mlpack;
using namespace mlpack::distribution;
using namespace mlpack::regression;
using namespace mlpack::bound;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::perceptron;
using namespace mlpack::regression;
using namespace mlpack::naive_bayes;
using namespace mlpack::neighbor;
using namespace mlpack::decision_stump;

using namespace arma;
using namespace boost;
using namespace boost::archive;
using namespace boost::serialization;
using namespace std;

BOOST_AUTO_TEST_SUITE(SerializationTest);

/**
 * Serialize a random cube.
 */
BOOST_AUTO_TEST_CASE(CubeSerializeTest)
{
  arma::cube m;
  m.randu(2, 50, 50);
  TestAllArmadilloSerialization(m);
}

/**
 * Serialize an empty cube.
 */
BOOST_AUTO_TEST_CASE(EmptyCubeSerializeTest)
{
  arma::cube c;
  TestAllArmadilloSerialization(c);
}


/**
 * Can we load and save an Armadillo matrix?
 */
BOOST_AUTO_TEST_CASE(MatrixSerializeXMLTest)
{
  arma::mat m;
  m.randu(50, 50);
  TestAllArmadilloSerialization(m);
}

/**
 * How about columns?
 */
BOOST_AUTO_TEST_CASE(ColSerializeTest)
{
  arma::vec m;
  m.randu(50, 1);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
BOOST_AUTO_TEST_CASE(RowSerializeTest)
{
  arma::rowvec m;
  m.randu(1, 50);
  TestAllArmadilloSerialization(m);
}

// A quick test with an empty matrix.
BOOST_AUTO_TEST_CASE(EmptyMatrixSerializeTest)
{
  arma::mat m;
  TestAllArmadilloSerialization(m);
}

/**
 * Can we load and save a sparse Armadillo matrix?
 */
BOOST_AUTO_TEST_CASE(SparseMatrixSerializeTest)
{
  arma::sp_mat m;
  m.sprandu(50, 50, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about columns?
 */
BOOST_AUTO_TEST_CASE(SparseColSerializeTest)
{
  arma::sp_vec m;
  m.sprandu(50, 1, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
BOOST_AUTO_TEST_CASE(SparseRowSerializeTest)
{
  arma::sp_rowvec m;
  m.sprandu(1, 50, 0.3);
  TestAllArmadilloSerialization(m);
}

// A quick test with an empty matrix.
BOOST_AUTO_TEST_CASE(EmptySparseMatrixSerializeTest)
{
  arma::sp_mat m;
  TestAllArmadilloSerialization(m);
}

// Now, test mlpack objects.
BOOST_AUTO_TEST_CASE(DiscreteDistributionTest)
{
  // I assume that I am properly saving vectors, so, this should be
  // straightforward.
  vec prob;
  prob.randu(12);
  DiscreteDistribution t(prob);

  DiscreteDistribution xmlT, textT, binaryT;

  // Load and save with all serializers.
  SerializeObjectAll(t, xmlT, textT, binaryT);

  for (size_t i = 0; i < 12; ++i)
  {
    vec obs(1);
    obs[0] = i;
    const double prob = t.Probability(obs);
    if (prob == 0.0)
    {
      BOOST_REQUIRE_SMALL(xmlT.Probability(obs), 1e-8);
      BOOST_REQUIRE_SMALL(textT.Probability(obs), 1e-8);
      BOOST_REQUIRE_SMALL(binaryT.Probability(obs), 1e-8);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(prob, xmlT.Probability(obs), 1e-8);
      BOOST_REQUIRE_CLOSE(prob, textT.Probability(obs), 1e-8);
      BOOST_REQUIRE_CLOSE(prob, binaryT.Probability(obs), 1e-8);
    }
  }
}

BOOST_AUTO_TEST_CASE(GaussianDistributionTest)
{
  vec mean(10);
  mean.randu();
  // Generate a covariance matrix.
  mat cov;
  cov.randu(10, 10);
  cov = (cov * cov.t());

  GaussianDistribution g(mean, cov);
  GaussianDistribution xmlG, textG, binaryG;

  SerializeObjectAll(g, xmlG, textG, binaryG);

  BOOST_REQUIRE_EQUAL(g.Dimensionality(), xmlG.Dimensionality());
  BOOST_REQUIRE_EQUAL(g.Dimensionality(), textG.Dimensionality());
  BOOST_REQUIRE_EQUAL(g.Dimensionality(), binaryG.Dimensionality());

  // First, check the means.
  CheckMatrices(g.Mean(), xmlG.Mean(), textG.Mean(), binaryG.Mean());

  // Now, check the covariance.
  CheckMatrices(g.Covariance(), xmlG.Covariance(), textG.Covariance(),
      binaryG.Covariance());

  // Lastly, run some observations through and make sure the probability is the
  // same.  This should test anything cached internally.
  arma::mat randomObs;
  randomObs.randu(10, 500);

  for (size_t i = 0; i < 500; ++i)
  {
    const double prob = g.Probability(randomObs.unsafe_col(i));

    if (prob == 0.0)
    {
      BOOST_REQUIRE_SMALL(xmlG.Probability(randomObs.unsafe_col(i)), 1e-8);
      BOOST_REQUIRE_SMALL(textG.Probability(randomObs.unsafe_col(i)), 1e-8);
      BOOST_REQUIRE_SMALL(binaryG.Probability(randomObs.unsafe_col(i)), 1e-8);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(prob, xmlG.Probability(randomObs.unsafe_col(i)),
          1e-8);
      BOOST_REQUIRE_CLOSE(prob, textG.Probability(randomObs.unsafe_col(i)),
          1e-8);
      BOOST_REQUIRE_CLOSE(prob, binaryG.Probability(randomObs.unsafe_col(i)),
          1e-8);
    }
  }
}

BOOST_AUTO_TEST_CASE(LaplaceDistributionTest)
{
  vec mean(20);
  mean.randu();

  LaplaceDistribution l(mean, 2.5);
  LaplaceDistribution xmlL, textL, binaryL;

  SerializeObjectAll(l, xmlL, textL, binaryL);

  BOOST_REQUIRE_CLOSE(l.Scale(), xmlL.Scale(), 1e-8);
  BOOST_REQUIRE_CLOSE(l.Scale(), textL.Scale(), 1e-8);
  BOOST_REQUIRE_CLOSE(l.Scale(), binaryL.Scale(), 1e-8);

  CheckMatrices(l.Mean(), xmlL.Mean(), textL.Mean(), binaryL.Mean());
}

BOOST_AUTO_TEST_CASE(MahalanobisDistanceTest)
{
  MahalanobisDistance<> d;
  d.Covariance().randu(50, 50);

  MahalanobisDistance<> xmlD, textD, binaryD;

  SerializeObjectAll(d, xmlD, textD, binaryD);

  // Check the covariance matrices.
  CheckMatrices(d.Covariance(),
                xmlD.Covariance(),
                textD.Covariance(),
                binaryD.Covariance());
}

BOOST_AUTO_TEST_CASE(LinearRegressionTest)
{
  // Generate some random data.
  mat data;
  data.randn(15, 800);
  vec responses;
  responses.randn(800, 1);

  LinearRegression lr(data, responses, 0.05); // Train the model.
  LinearRegression xmlLr, textLr, binaryLr;

  SerializeObjectAll(lr, xmlLr, textLr, binaryLr);

  BOOST_REQUIRE_CLOSE(lr.Lambda(), xmlLr.Lambda(), 1e-8);
  BOOST_REQUIRE_CLOSE(lr.Lambda(), textLr.Lambda(), 1e-8);
  BOOST_REQUIRE_CLOSE(lr.Lambda(), binaryLr.Lambda(), 1e-8);

  CheckMatrices(lr.Parameters(), xmlLr.Parameters(), textLr.Parameters(),
      binaryLr.Parameters());
}

BOOST_AUTO_TEST_CASE(RegressionDistributionTest)
{
  // Generate some random data.
  mat data;
  data.randn(15, 800);
  vec responses;
  responses.randn(800, 1);

  RegressionDistribution rd(data, responses);
  RegressionDistribution xmlRd, textRd, binaryRd;

  // Okay, now save it and load it.
  SerializeObjectAll(rd, xmlRd, textRd, binaryRd);

  // Check the gaussian distribution.
  CheckMatrices(rd.Err().Mean(),
                xmlRd.Err().Mean(),
                textRd.Err().Mean(),
                binaryRd.Err().Mean());
  CheckMatrices(rd.Err().Covariance(),
                xmlRd.Err().Covariance(),
                textRd.Err().Covariance(),
                binaryRd.Err().Covariance());

  // Check the regression function.
  if (rd.Rf().Lambda() == 0.0)
  {
    BOOST_REQUIRE_SMALL(xmlRd.Rf().Lambda(), 1e-8);
    BOOST_REQUIRE_SMALL(textRd.Rf().Lambda(), 1e-8);
    BOOST_REQUIRE_SMALL(binaryRd.Rf().Lambda(), 1e-8);
  }
  else
  {
    BOOST_REQUIRE_CLOSE(rd.Rf().Lambda(), xmlRd.Rf().Lambda(), 1e-8);
    BOOST_REQUIRE_CLOSE(rd.Rf().Lambda(), textRd.Rf().Lambda(), 1e-8);
    BOOST_REQUIRE_CLOSE(rd.Rf().Lambda(), binaryRd.Rf().Lambda(), 1e-8);
  }

  CheckMatrices(rd.Rf().Parameters(),
                xmlRd.Rf().Parameters(),
                textRd.Rf().Parameters(),
                binaryRd.Rf().Parameters());
}

BOOST_AUTO_TEST_CASE(BallBoundTest)
{
  BallBound<> b(100);
  b.Center().randu();
  b.Radius() = 14.0;

  BallBound<> xmlB, textB, binaryB;

  SerializeObjectAll(b, xmlB, textB, binaryB);

  // Check the dimensionality.
  BOOST_REQUIRE_EQUAL(b.Dim(), xmlB.Dim());
  BOOST_REQUIRE_EQUAL(b.Dim(), textB.Dim());
  BOOST_REQUIRE_EQUAL(b.Dim(), binaryB.Dim());

  // Check the radius.
  BOOST_REQUIRE_CLOSE(b.Radius(), xmlB.Radius(), 1e-8);
  BOOST_REQUIRE_CLOSE(b.Radius(), textB.Radius(), 1e-8);
  BOOST_REQUIRE_CLOSE(b.Radius(), binaryB.Radius(), 1e-8);

  // Now check the vectors.
  CheckMatrices(b.Center(), xmlB.Center(), textB.Center(), binaryB.Center());
}

BOOST_AUTO_TEST_CASE(MahalanobisBallBoundTest)
{
  BallBound<MahalanobisDistance<>, arma::vec> b(100);
  b.Center().randu();
  b.Radius() = 14.0;
  b.Metric().Covariance().randu(100, 100);

  BallBound<MahalanobisDistance<>, arma::vec> xmlB, textB, binaryB;

  SerializeObjectAll(b, xmlB, textB, binaryB);

  // Check the radius.
  BOOST_REQUIRE_CLOSE(b.Radius(), xmlB.Radius(), 1e-8);
  BOOST_REQUIRE_CLOSE(b.Radius(), textB.Radius(), 1e-8);
  BOOST_REQUIRE_CLOSE(b.Radius(), binaryB.Radius(), 1e-8);

  // Check the vectors.
  CheckMatrices(b.Center(), xmlB.Center(), textB.Center(), binaryB.Center());
  CheckMatrices(b.Metric().Covariance(),
                xmlB.Metric().Covariance(),
                textB.Metric().Covariance(),
                binaryB.Metric().Covariance());
}

BOOST_AUTO_TEST_CASE(HRectBoundTest)
{
  HRectBound<> b(2);

  arma::mat points("0.0, 1.1; 5.0, 2.2");
  points = points.t();
  b |= points; // [0.0, 5.0]; [1.1, 2.2];

  HRectBound<> xmlB, textB, binaryB;

  SerializeObjectAll(b, xmlB, textB, binaryB);

  // Check the dimensionality.
  BOOST_REQUIRE_EQUAL(b.Dim(), xmlB.Dim());
  BOOST_REQUIRE_EQUAL(b.Dim(), textB.Dim());
  BOOST_REQUIRE_EQUAL(b.Dim(), binaryB.Dim());

  // Check the bounds.
  for (size_t i = 0; i < b.Dim(); ++i)
  {
    BOOST_REQUIRE_CLOSE(b[i].Lo(), xmlB[i].Lo(), 1e-8);
    BOOST_REQUIRE_CLOSE(b[i].Hi(), xmlB[i].Hi(), 1e-8);
    BOOST_REQUIRE_CLOSE(b[i].Lo(), textB[i].Lo(), 1e-8);
    BOOST_REQUIRE_CLOSE(b[i].Hi(), textB[i].Hi(), 1e-8);
    BOOST_REQUIRE_CLOSE(b[i].Lo(), binaryB[i].Lo(), 1e-8);
    BOOST_REQUIRE_CLOSE(b[i].Hi(), binaryB[i].Hi(), 1e-8);
  }

  // Check the minimum width.
  BOOST_REQUIRE_CLOSE(b.MinWidth(), xmlB.MinWidth(), 1e-8);
  BOOST_REQUIRE_CLOSE(b.MinWidth(), textB.MinWidth(), 1e-8);
  BOOST_REQUIRE_CLOSE(b.MinWidth(), binaryB.MinWidth(), 1e-8);
}

template<typename TreeType>
void CheckTrees(TreeType& tree,
                TreeType& xmlTree,
                TreeType& textTree,
                TreeType& binaryTree)
{
  const typename TreeType::Mat* dataset = &tree.Dataset();

  // Make sure that the data matrices are the same.
  if (tree.Parent() == NULL)
  {
    CheckMatrices(*dataset,
                  xmlTree.Dataset(),
                  textTree.Dataset(),
                  binaryTree.Dataset());

    // Also ensure that the other parents are null too.
    BOOST_REQUIRE_EQUAL(xmlTree.Parent(), (TreeType*) NULL);
    BOOST_REQUIRE_EQUAL(textTree.Parent(), (TreeType*) NULL);
    BOOST_REQUIRE_EQUAL(binaryTree.Parent(), (TreeType*) NULL);
  }

  // Make sure the number of children is the same.
  BOOST_REQUIRE_EQUAL(tree.NumChildren(), xmlTree.NumChildren());
  BOOST_REQUIRE_EQUAL(tree.NumChildren(), textTree.NumChildren());
  BOOST_REQUIRE_EQUAL(tree.NumChildren(), binaryTree.NumChildren());

  // Make sure the number of descendants is the same.
  BOOST_REQUIRE_EQUAL(tree.NumDescendants(), xmlTree.NumDescendants());
  BOOST_REQUIRE_EQUAL(tree.NumDescendants(), textTree.NumDescendants());
  BOOST_REQUIRE_EQUAL(tree.NumDescendants(), binaryTree.NumDescendants());

  // Make sure the number of points is the same.
  BOOST_REQUIRE_EQUAL(tree.NumPoints(), xmlTree.NumPoints());
  BOOST_REQUIRE_EQUAL(tree.NumPoints(), textTree.NumPoints());
  BOOST_REQUIRE_EQUAL(tree.NumPoints(), binaryTree.NumPoints());

  // Check that each point is the same.
  for (size_t i = 0; i < tree.NumPoints(); ++i)
  {
    BOOST_REQUIRE_EQUAL(tree.Point(i), xmlTree.Point(i));
    BOOST_REQUIRE_EQUAL(tree.Point(i), textTree.Point(i));
    BOOST_REQUIRE_EQUAL(tree.Point(i), binaryTree.Point(i));
  }

  // Check that the parent distance is the same.
  BOOST_REQUIRE_CLOSE(tree.ParentDistance(), xmlTree.ParentDistance(), 1e-8);
  BOOST_REQUIRE_CLOSE(tree.ParentDistance(), textTree.ParentDistance(), 1e-8);
  BOOST_REQUIRE_CLOSE(tree.ParentDistance(), binaryTree.ParentDistance(), 1e-8);

  // Check that the furthest descendant distance is the same.
  BOOST_REQUIRE_CLOSE(tree.FurthestDescendantDistance(),
      xmlTree.FurthestDescendantDistance(), 1e-8);
  BOOST_REQUIRE_CLOSE(tree.FurthestDescendantDistance(),
      textTree.FurthestDescendantDistance(), 1e-8);
  BOOST_REQUIRE_CLOSE(tree.FurthestDescendantDistance(),
      binaryTree.FurthestDescendantDistance(), 1e-8);

  // Check that the minimum bound distance is the same.
  BOOST_REQUIRE_CLOSE(tree.MinimumBoundDistance(),
      xmlTree.MinimumBoundDistance(), 1e-8);
  BOOST_REQUIRE_CLOSE(tree.MinimumBoundDistance(),
      textTree.MinimumBoundDistance(), 1e-8);
  BOOST_REQUIRE_CLOSE(tree.MinimumBoundDistance(),
      binaryTree.MinimumBoundDistance(), 1e-8);

  // Recurse into the children.
  for (size_t i = 0; i < tree.NumChildren(); ++i)
  {
    // Check that the child dataset is the same.
    BOOST_REQUIRE_EQUAL(&xmlTree.Dataset(), &xmlTree.Child(i).Dataset());
    BOOST_REQUIRE_EQUAL(&textTree.Dataset(), &textTree.Child(i).Dataset());
    BOOST_REQUIRE_EQUAL(&binaryTree.Dataset(), &binaryTree.Child(i).Dataset());

    // Make sure the parent link is right.
    BOOST_REQUIRE_EQUAL(xmlTree.Child(i).Parent(), &xmlTree);
    BOOST_REQUIRE_EQUAL(textTree.Child(i).Parent(), &textTree);
    BOOST_REQUIRE_EQUAL(binaryTree.Child(i).Parent(), &binaryTree);

    CheckTrees(tree.Child(i), xmlTree.Child(i), textTree.Child(i),
        binaryTree.Child(i));
  }
}

BOOST_AUTO_TEST_CASE(BinarySpaceTreeTest)
{
  arma::mat data;
  data.randu(3, 100);
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  TreeType* xmlTree;
  TreeType* textTree;
  TreeType* binaryTree;

  SerializePointerObjectAll(&tree, xmlTree, textTree, binaryTree);

  CheckTrees(tree, *xmlTree, *textTree, *binaryTree);

  delete xmlTree;
  delete textTree;
  delete binaryTree;
}

BOOST_AUTO_TEST_CASE(BinarySpaceTreeOverwriteTest)
{
  arma::mat data;
  data.randu(3, 100);
  typedef KDTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  arma::mat otherData;
  otherData.randu(5, 50);
  TreeType xmlTree(otherData);
  TreeType textTree(xmlTree);
  TreeType binaryTree(xmlTree);

  SerializeObjectAll(tree, xmlTree, textTree, binaryTree);

  CheckTrees(tree, xmlTree, textTree, binaryTree);
}

BOOST_AUTO_TEST_CASE(CoverTreeTest)
{
  arma::mat data;
  data.randu(3, 100);
  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType tree(data);

  TreeType* xmlTree;
  TreeType* textTree;
  TreeType* binaryTree;

  SerializePointerObjectAll(&tree, xmlTree, textTree, binaryTree);

  CheckTrees(tree, *xmlTree, *textTree, *binaryTree);

  // Also check a few other things.
  std::stack<TreeType*> stack, xmlStack, textStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(xmlTree);
  textStack.push(textTree);
  binaryStack.push(binaryTree);
  while (!stack.empty())
  {
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* textNode = textStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    textStack.pop();
    binaryStack.pop();

    BOOST_REQUIRE_EQUAL(node->Scale(), xmlNode->Scale());
    BOOST_REQUIRE_EQUAL(node->Scale(), textNode->Scale());
    BOOST_REQUIRE_EQUAL(node->Scale(), binaryNode->Scale());

    BOOST_REQUIRE_CLOSE(node->Base(), xmlNode->Base(), 1e-5);
    BOOST_REQUIRE_CLOSE(node->Base(), textNode->Base(), 1e-5);
    BOOST_REQUIRE_CLOSE(node->Base(), binaryNode->Base(), 1e-5);

    for (size_t i = 0; i < node->NumChildren(); ++i)
    {
      stack.push(&node->Child(i));
      xmlStack.push(&xmlNode->Child(i));
      textStack.push(&textNode->Child(i));
      binaryStack.push(&binaryNode->Child(i));
    }
  }

  delete xmlTree;
  delete textTree;
  delete binaryTree;
}

BOOST_AUTO_TEST_CASE(CoverTreeOverwriteTest)
{
  arma::mat data;
  data.randu(3, 100);
  typedef StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat>
      TreeType;
  TreeType tree(data);

  arma::mat otherData;
  otherData.randu(5, 50);
  TreeType xmlTree(otherData);
  TreeType textTree(xmlTree);
  TreeType binaryTree(xmlTree);

  SerializeObjectAll(tree, xmlTree, textTree, binaryTree);

  CheckTrees(tree, xmlTree, textTree, binaryTree);

  // Also check a few other things.
  std::stack<TreeType*> stack, xmlStack, textStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(&xmlTree);
  textStack.push(&textTree);
  binaryStack.push(&binaryTree);
  while (!stack.empty())
  {
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* textNode = textStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    textStack.pop();
    binaryStack.pop();

    BOOST_REQUIRE_EQUAL(node->Scale(), xmlNode->Scale());
    BOOST_REQUIRE_EQUAL(node->Scale(), textNode->Scale());
    BOOST_REQUIRE_EQUAL(node->Scale(), binaryNode->Scale());

    BOOST_REQUIRE_CLOSE(node->Base(), xmlNode->Base(), 1e-5);
    BOOST_REQUIRE_CLOSE(node->Base(), textNode->Base(), 1e-5);
    BOOST_REQUIRE_CLOSE(node->Base(), binaryNode->Base(), 1e-5);

    for (size_t i = 0; i < node->NumChildren(); ++i)
    {
      stack.push(&node->Child(i));
      xmlStack.push(&xmlNode->Child(i));
      textStack.push(&textNode->Child(i));
      binaryStack.push(&binaryNode->Child(i));
    }
  }
}

BOOST_AUTO_TEST_CASE(RectangleTreeTest)
{
  arma::mat data;
  data.randu(3, 1000);
  typedef RTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  TreeType* xmlTree;
  TreeType* textTree;
  TreeType* binaryTree;

  SerializePointerObjectAll(&tree, xmlTree, textTree, binaryTree);

  CheckTrees(tree, *xmlTree, *textTree, *binaryTree);

  // Check a few other things too.
  std::stack<TreeType*> stack, xmlStack, textStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(xmlTree);
  textStack.push(textTree);
  binaryStack.push(binaryTree);
  while (!stack.empty())
  {
    // Check more things...
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* textNode = textStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    textStack.pop();
    binaryStack.pop();

    BOOST_REQUIRE_EQUAL(node->MaxLeafSize(), xmlNode->MaxLeafSize());
    BOOST_REQUIRE_EQUAL(node->MaxLeafSize(), textNode->MaxLeafSize());
    BOOST_REQUIRE_EQUAL(node->MaxLeafSize(), binaryNode->MaxLeafSize());

    BOOST_REQUIRE_EQUAL(node->MinLeafSize(), xmlNode->MinLeafSize());
    BOOST_REQUIRE_EQUAL(node->MinLeafSize(), textNode->MinLeafSize());
    BOOST_REQUIRE_EQUAL(node->MinLeafSize(), binaryNode->MinLeafSize());

    BOOST_REQUIRE_EQUAL(node->MaxNumChildren(), xmlNode->MaxNumChildren());
    BOOST_REQUIRE_EQUAL(node->MaxNumChildren(), textNode->MaxNumChildren());
    BOOST_REQUIRE_EQUAL(node->MaxNumChildren(), binaryNode->MaxNumChildren());

    BOOST_REQUIRE_EQUAL(node->MinNumChildren(), xmlNode->MinNumChildren());
    BOOST_REQUIRE_EQUAL(node->MinNumChildren(), textNode->MinNumChildren());
    BOOST_REQUIRE_EQUAL(node->MinNumChildren(), binaryNode->MinNumChildren());
  }

  delete xmlTree;
  delete textTree;
  delete binaryTree;
}

BOOST_AUTO_TEST_CASE(RectangleTreeOverwriteTest)
{
  arma::mat data;
  data.randu(3, 1000);
  typedef RTree<EuclideanDistance, EmptyStatistic, arma::mat> TreeType;
  TreeType tree(data);

  arma::mat otherData;
  otherData.randu(5, 50);
  TreeType xmlTree(otherData);
  TreeType textTree(otherData);
  TreeType binaryTree(textTree);

  SerializeObjectAll(tree, xmlTree, textTree, binaryTree);

  CheckTrees(tree, xmlTree, textTree, binaryTree);

  // Check a few other things too.
  std::stack<TreeType*> stack, xmlStack, textStack, binaryStack;
  stack.push(&tree);
  xmlStack.push(&xmlTree);
  textStack.push(&textTree);
  binaryStack.push(&binaryTree);
  while (!stack.empty())
  {
    // Check more things...
    TreeType* node = stack.top();
    TreeType* xmlNode = xmlStack.top();
    TreeType* textNode = textStack.top();
    TreeType* binaryNode = binaryStack.top();
    stack.pop();
    xmlStack.pop();
    textStack.pop();
    binaryStack.pop();

    BOOST_REQUIRE_EQUAL(node->MaxLeafSize(), xmlNode->MaxLeafSize());
    BOOST_REQUIRE_EQUAL(node->MaxLeafSize(), textNode->MaxLeafSize());
    BOOST_REQUIRE_EQUAL(node->MaxLeafSize(), binaryNode->MaxLeafSize());

    BOOST_REQUIRE_EQUAL(node->MinLeafSize(), xmlNode->MinLeafSize());
    BOOST_REQUIRE_EQUAL(node->MinLeafSize(), textNode->MinLeafSize());
    BOOST_REQUIRE_EQUAL(node->MinLeafSize(), binaryNode->MinLeafSize());

    BOOST_REQUIRE_EQUAL(node->MaxNumChildren(), xmlNode->MaxNumChildren());
    BOOST_REQUIRE_EQUAL(node->MaxNumChildren(), textNode->MaxNumChildren());
    BOOST_REQUIRE_EQUAL(node->MaxNumChildren(), binaryNode->MaxNumChildren());

    BOOST_REQUIRE_EQUAL(node->MinNumChildren(), xmlNode->MinNumChildren());
    BOOST_REQUIRE_EQUAL(node->MinNumChildren(), textNode->MinNumChildren());
    BOOST_REQUIRE_EQUAL(node->MinNumChildren(), binaryNode->MinNumChildren());
  }
}

BOOST_AUTO_TEST_CASE(PerceptronTest)
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

  BOOST_REQUIRE_EQUAL(p.MaxIterations(), pXml.MaxIterations());
  BOOST_REQUIRE_EQUAL(p.MaxIterations(), pText.MaxIterations());
  BOOST_REQUIRE_EQUAL(p.MaxIterations(), pBinary.MaxIterations());
}

BOOST_AUTO_TEST_CASE(LogisticRegressionTest)
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

  BOOST_REQUIRE_CLOSE(lr.Lambda(), lrXml.Lambda(), 1e-5);
  BOOST_REQUIRE_CLOSE(lr.Lambda(), lrText.Lambda(), 1e-5);
  BOOST_REQUIRE_CLOSE(lr.Lambda(), lrBinary.Lambda(), 1e-5);
}

BOOST_AUTO_TEST_CASE(KNNTest)
{
  using neighbor::KNN;
  arma::mat dataset = arma::randu<arma::mat>(5, 2000);

  KNN knn(dataset, DUAL_TREE_MODE);

  KNN knnXml, knnText, knnBinary;

  SerializeObjectAll(knn, knnXml, knnText, knnBinary);

  // Now run nearest neighbor and make sure the results are the same.
  arma::mat querySet = arma::randu<arma::mat>(5, 1000);

  arma::mat distances, xmlDistances, textDistances, binaryDistances;
  arma::Mat<size_t> neighbors, xmlNeighbors, textNeighbors, binaryNeighbors;

  knn.Search(querySet, 5, neighbors, distances);
  knnXml.Search(querySet, 5, xmlNeighbors, xmlDistances);
  knnText.Search(querySet, 5, textNeighbors, textDistances);
  knnBinary.Search(querySet, 5, binaryNeighbors, binaryDistances);

  CheckMatrices(distances, xmlDistances, textDistances, binaryDistances);
  CheckMatrices(neighbors, xmlNeighbors, textNeighbors, binaryNeighbors);
}

BOOST_AUTO_TEST_CASE(SoftmaxRegressionTest)
{
  using regression::SoftmaxRegression;

  arma::mat dataset = arma::randu<arma::mat>(5, 1000);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 500; ++i)
    labels[i] = 0;
  for (size_t i = 500; i < 1000; ++i)
    labels[i] = 1;

  SoftmaxRegression<> sr(dataset, labels, 2);

  SoftmaxRegression<> srXml(dataset.n_rows, 2);
  SoftmaxRegression<> srText(dataset.n_rows, 2);
  SoftmaxRegression<> srBinary(dataset.n_rows, 2);

  SerializeObjectAll(sr, srXml, srText, srBinary);

  CheckMatrices(sr.Parameters(), srXml.Parameters(), srText.Parameters(),
      srBinary.Parameters());
}

BOOST_AUTO_TEST_CASE(DETTest)
{
  using det::DTree;
  typedef DTree<arma::mat>   DTreeX;

  // Create a density estimation tree on a random dataset.
  arma::mat dataset = arma::randu<arma::mat>(25, 5000);

  DTreeX tree(dataset);

  arma::mat otherDataset = arma::randu<arma::mat>(5, 100);
  DTreeX xmlTree, binaryTree, textTree(otherDataset);

  SerializeObjectAll(tree, xmlTree, binaryTree, textTree);

  std::stack<DTreeX*> stack, xmlStack, binaryStack, textStack;
  stack.push(&tree);
  xmlStack.push(&xmlTree);
  binaryStack.push(&binaryTree);
  textStack.push(&textTree);

  while (!stack.empty())
  {
    // Get the top node from the stack.
    DTreeX* node = stack.top();
    DTreeX* xmlNode = xmlStack.top();
    DTreeX* binaryNode = binaryStack.top();
    DTreeX* textNode = textStack.top();

    stack.pop();
    xmlStack.pop();
    binaryStack.pop();
    textStack.pop();

    // Check that all the members are the same.
    BOOST_REQUIRE_EQUAL(node->Start(), xmlNode->Start());
    BOOST_REQUIRE_EQUAL(node->Start(), binaryNode->Start());
    BOOST_REQUIRE_EQUAL(node->Start(), textNode->Start());

    BOOST_REQUIRE_EQUAL(node->End(), xmlNode->End());
    BOOST_REQUIRE_EQUAL(node->End(), binaryNode->End());
    BOOST_REQUIRE_EQUAL(node->End(), textNode->End());

    BOOST_REQUIRE_EQUAL(node->SplitDim(), xmlNode->SplitDim());
    BOOST_REQUIRE_EQUAL(node->SplitDim(), binaryNode->SplitDim());
    BOOST_REQUIRE_EQUAL(node->SplitDim(), textNode->SplitDim());

    if (std::abs(node->SplitValue()) < 1e-5)
    {
      BOOST_REQUIRE_SMALL(xmlNode->SplitValue(), 1e-5);
      BOOST_REQUIRE_SMALL(binaryNode->SplitValue(), 1e-5);
      BOOST_REQUIRE_SMALL(textNode->SplitValue(), 1e-5);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(node->SplitValue(), xmlNode->SplitValue(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->SplitValue(), binaryNode->SplitValue(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->SplitValue(), textNode->SplitValue(), 1e-5);
    }

    if (std::abs(node->LogNegError()) < 1e-5)
    {
      BOOST_REQUIRE_SMALL(xmlNode->LogNegError(), 1e-5);
      BOOST_REQUIRE_SMALL(binaryNode->LogNegError(), 1e-5);
      BOOST_REQUIRE_SMALL(textNode->LogNegError(), 1e-5);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(node->LogNegError(), xmlNode->LogNegError(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->LogNegError(), binaryNode->LogNegError(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->LogNegError(), textNode->LogNegError(), 1e-5);
    }

    if (std::abs(node->SubtreeLeavesLogNegError()) < 1e-5)
    {
      BOOST_REQUIRE_SMALL(xmlNode->SubtreeLeavesLogNegError(), 1e-5);
      BOOST_REQUIRE_SMALL(binaryNode->SubtreeLeavesLogNegError(), 1e-5);
      BOOST_REQUIRE_SMALL(textNode->SubtreeLeavesLogNegError(), 1e-5);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(node->SubtreeLeavesLogNegError(),
          xmlNode->SubtreeLeavesLogNegError(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->SubtreeLeavesLogNegError(),
          binaryNode->SubtreeLeavesLogNegError(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->SubtreeLeavesLogNegError(),
          textNode->SubtreeLeavesLogNegError(), 1e-5);
    }

    BOOST_REQUIRE_EQUAL(node->SubtreeLeaves(), xmlNode->SubtreeLeaves());
    BOOST_REQUIRE_EQUAL(node->SubtreeLeaves(), binaryNode->SubtreeLeaves());
    BOOST_REQUIRE_EQUAL(node->SubtreeLeaves(), textNode->SubtreeLeaves());

    if (std::abs(node->Ratio()) < 1e-5)
    {
      BOOST_REQUIRE_SMALL(xmlNode->Ratio(), 1e-5);
      BOOST_REQUIRE_SMALL(binaryNode->Ratio(), 1e-5);
      BOOST_REQUIRE_SMALL(textNode->Ratio(), 1e-5);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(node->Ratio(), xmlNode->Ratio(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->Ratio(), binaryNode->Ratio(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->Ratio(), textNode->Ratio(), 1e-5);
    }

    if (std::abs(node->LogVolume()) < 1e-5)
    {
      BOOST_REQUIRE_SMALL(xmlNode->LogVolume(), 1e-5);
      BOOST_REQUIRE_SMALL(binaryNode->LogVolume(), 1e-5);
      BOOST_REQUIRE_SMALL(textNode->LogVolume(), 1e-5);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(node->LogVolume(), xmlNode->LogVolume(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->LogVolume(), binaryNode->LogVolume(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->LogVolume(), textNode->LogVolume(), 1e-5);
    }

    if (node->Left() == NULL)
    {
      BOOST_REQUIRE(xmlNode->Left() == NULL);
      BOOST_REQUIRE(binaryNode->Left() == NULL);
      BOOST_REQUIRE(textNode->Left() == NULL);
    }
    else
    {
      BOOST_REQUIRE(xmlNode->Left() != NULL);
      BOOST_REQUIRE(binaryNode->Left() != NULL);
      BOOST_REQUIRE(textNode->Left() != NULL);

      // Push children onto stack.
      stack.push(node->Left());
      xmlStack.push(xmlNode->Left());
      binaryStack.push(binaryNode->Left());
      textStack.push(textNode->Left());
    }

    if (node->Right() == NULL)
    {
      BOOST_REQUIRE(xmlNode->Right() == NULL);
      BOOST_REQUIRE(binaryNode->Right() == NULL);
      BOOST_REQUIRE(textNode->Right() == NULL);
    }
    else
    {
      BOOST_REQUIRE(xmlNode->Right() != NULL);
      BOOST_REQUIRE(binaryNode->Right() != NULL);
      BOOST_REQUIRE(textNode->Right() != NULL);

      // Push children onto stack.
      stack.push(node->Right());
      xmlStack.push(xmlNode->Right());
      binaryStack.push(binaryNode->Right());
      textStack.push(textNode->Right());
    }

    BOOST_REQUIRE_EQUAL(node->Root(), xmlNode->Root());
    BOOST_REQUIRE_EQUAL(node->Root(), binaryNode->Root());
    BOOST_REQUIRE_EQUAL(node->Root(), textNode->Root());

    if (std::abs(node->AlphaUpper()) < 1e-5)
    {
      BOOST_REQUIRE_SMALL(xmlNode->AlphaUpper(), 1e-5);
      BOOST_REQUIRE_SMALL(binaryNode->AlphaUpper(), 1e-5);
      BOOST_REQUIRE_SMALL(textNode->AlphaUpper(), 1e-5);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(node->AlphaUpper(), xmlNode->AlphaUpper(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->AlphaUpper(), binaryNode->AlphaUpper(), 1e-5);
      BOOST_REQUIRE_CLOSE(node->AlphaUpper(), textNode->AlphaUpper(), 1e-5);
    }

    BOOST_REQUIRE_EQUAL(node->MaxVals().n_elem, xmlNode->MaxVals().n_elem);
    BOOST_REQUIRE_EQUAL(node->MaxVals().n_elem, binaryNode->MaxVals().n_elem);
    BOOST_REQUIRE_EQUAL(node->MaxVals().n_elem, textNode->MaxVals().n_elem);
    for (size_t i = 0; i < node->MaxVals().n_elem; ++i)
    {
      if (std::abs(node->MaxVals()[i]) < 1e-5)
      {
        BOOST_REQUIRE_SMALL(xmlNode->MaxVals()[i], 1e-5);
        BOOST_REQUIRE_SMALL(binaryNode->MaxVals()[i], 1e-5);
        BOOST_REQUIRE_SMALL(textNode->MaxVals()[i], 1e-5);
      }
      else
      {
        BOOST_REQUIRE_CLOSE(node->MaxVals()[i], xmlNode->MaxVals()[i], 1e-5);
        BOOST_REQUIRE_CLOSE(node->MaxVals()[i], binaryNode->MaxVals()[i], 1e-5);
        BOOST_REQUIRE_CLOSE(node->MaxVals()[i], textNode->MaxVals()[i], 1e-5);
      }
    }

    BOOST_REQUIRE_EQUAL(node->MinVals().n_elem, xmlNode->MinVals().n_elem);
    BOOST_REQUIRE_EQUAL(node->MinVals().n_elem, binaryNode->MinVals().n_elem);
    BOOST_REQUIRE_EQUAL(node->MinVals().n_elem, textNode->MinVals().n_elem);
    for (size_t i = 0; i < node->MinVals().n_elem; ++i)
    {
      if (std::abs(node->MinVals()[i]) < 1e-5)
      {
        BOOST_REQUIRE_SMALL(xmlNode->MinVals()[i], 1e-5);
        BOOST_REQUIRE_SMALL(binaryNode->MinVals()[i], 1e-5);
        BOOST_REQUIRE_SMALL(textNode->MinVals()[i], 1e-5);
      }
      else
      {
        BOOST_REQUIRE_CLOSE(node->MinVals()[i], xmlNode->MinVals()[i], 1e-5);
        BOOST_REQUIRE_CLOSE(node->MinVals()[i], binaryNode->MinVals()[i], 1e-5);
        BOOST_REQUIRE_CLOSE(node->MinVals()[i], textNode->MinVals()[i], 1e-5);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(NaiveBayesSerializationTest)
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
  NaiveBayesClassifier<> xmlNbc(0, 0), textNbc(0, 0), binaryNbc(0, 0);
  SerializeObjectAll(nbc, xmlNbc, textNbc, binaryNbc);

  BOOST_REQUIRE_EQUAL(nbc.Means().n_elem, xmlNbc.Means().n_elem);
  BOOST_REQUIRE_EQUAL(nbc.Means().n_elem, textNbc.Means().n_elem);
  BOOST_REQUIRE_EQUAL(nbc.Means().n_elem, binaryNbc.Means().n_elem);
  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    BOOST_REQUIRE_CLOSE(nbc.Means()[i], xmlNbc.Means()[i], 1e-5);
    BOOST_REQUIRE_CLOSE(nbc.Means()[i], textNbc.Means()[i], 1e-5);
    BOOST_REQUIRE_CLOSE(nbc.Means()[i], binaryNbc.Means()[i], 1e-5);
  }

  BOOST_REQUIRE_EQUAL(nbc.Variances().n_elem, xmlNbc.Variances().n_elem);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_elem, textNbc.Variances().n_elem);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_elem, binaryNbc.Variances().n_elem);
  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    BOOST_REQUIRE_CLOSE(nbc.Variances()[i], xmlNbc.Variances()[i], 1e-5);
    BOOST_REQUIRE_CLOSE(nbc.Variances()[i], textNbc.Variances()[i], 1e-5);
    BOOST_REQUIRE_CLOSE(nbc.Variances()[i], binaryNbc.Variances()[i], 1e-5);
  }

  BOOST_REQUIRE_EQUAL(nbc.Probabilities().n_elem,
      xmlNbc.Probabilities().n_elem);
  BOOST_REQUIRE_EQUAL(nbc.Probabilities().n_elem,
      textNbc.Probabilities().n_elem);
  BOOST_REQUIRE_EQUAL(nbc.Probabilities().n_elem,
      binaryNbc.Probabilities().n_elem);
  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    BOOST_REQUIRE_CLOSE(nbc.Probabilities()[i], xmlNbc.Probabilities()[i],
        1e-5);
    BOOST_REQUIRE_CLOSE(nbc.Probabilities()[i], textNbc.Probabilities()[i],
        1e-5);
    BOOST_REQUIRE_CLOSE(nbc.Probabilities()[i], binaryNbc.Probabilities()[i],
        1e-5);
  }
}

BOOST_AUTO_TEST_CASE(RASearchTest)
{
  using neighbor::AllkRANN;
  using neighbor::KNN;
  arma::mat dataset = arma::randu<arma::mat>(5, 200);
  arma::mat otherDataset = arma::randu<arma::mat>(5, 100);

  // Find nearest neighbors in the top 10, with accuracy 0.95.  So 95% of the
  // results we get (at least) should fall into the top 10 of the true nearest
  // neighbors.
  AllkRANN allkrann(dataset, false, false, 5, 0.95);

  AllkRANN krannXml(otherDataset, false, false);
  AllkRANN krannText(otherDataset, true, false);
  AllkRANN krannBinary(otherDataset, true, true);

  SerializeObjectAll(allkrann, krannXml, krannText, krannBinary);

  // Now run nearest neighbor and make sure the results are the same.
  arma::mat querySet = arma::randu<arma::mat>(5, 100);

  arma::mat distances, xmlDistances, textDistances, binaryDistances;
  arma::Mat<size_t> neighbors, xmlNeighbors, textNeighbors, binaryNeighbors;

  KNN knn(dataset); // Exact search.
  knn.Search(querySet, 10, neighbors, distances);
  krannXml.Search(querySet, 5, xmlNeighbors, xmlDistances);
  krannText.Search(querySet, 5, textNeighbors, textDistances);
  krannBinary.Search(querySet, 5, binaryNeighbors, binaryDistances);

  BOOST_REQUIRE_EQUAL(xmlNeighbors.n_rows, 5);
  BOOST_REQUIRE_EQUAL(xmlNeighbors.n_cols, 100);
  BOOST_REQUIRE_EQUAL(textNeighbors.n_rows, 5);
  BOOST_REQUIRE_EQUAL(textNeighbors.n_cols, 100);
  BOOST_REQUIRE_EQUAL(binaryNeighbors.n_rows, 5);
  BOOST_REQUIRE_EQUAL(binaryNeighbors.n_cols, 100);

  size_t xmlCorrect = 0;
  size_t textCorrect = 0;
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
        if (neighbors(k, i) == textNeighbors(j, i))
          textCorrect++;
        if (neighbors(k, i) == binaryNeighbors(j, i))
          binaryCorrect++;
      }
    }
  }

  // We need 95% of these to be correct.
  BOOST_REQUIRE_GT(xmlCorrect, 95 * 5);
  BOOST_REQUIRE_GT(binaryCorrect, 95 * 5);
  BOOST_REQUIRE_GT(textCorrect, 95 * 5);
}

/**
 * Test that an LSH model can be serialized and deserialized.
 */
BOOST_AUTO_TEST_CASE(LSHTest)
{
  // Since we still don't have good tests for LSH, basically what we're going to
  // do is serialize an LSH model, and make sure we can deserialize it and that
  // we still get results when we call Search().
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);

  LSHSearch<> lsh(referenceData, 5, 10); // Arbitrary chosen parameters.

  LSHSearch<> xmlLsh;
  arma::mat textData = arma::randu<arma::mat>(5, 50);
  LSHSearch<> textLsh(textData, 4, 5);
  LSHSearch<> binaryLsh(referenceData, 15, 2);

  // Now serialize.
  SerializeObjectAll(lsh, xmlLsh, textLsh, binaryLsh);

  // Check what we can about the serialized objects.
  BOOST_REQUIRE_EQUAL(lsh.NumProjections(), xmlLsh.NumProjections());
  BOOST_REQUIRE_EQUAL(lsh.NumProjections(), textLsh.NumProjections());
  BOOST_REQUIRE_EQUAL(lsh.NumProjections(), binaryLsh.NumProjections());
  for (size_t i = 0; i < lsh.NumProjections(); ++i)
  {
    CheckMatrices(lsh.Projections().slice(i), xmlLsh.Projections().slice(i),
        textLsh.Projections().slice(i), binaryLsh.Projections().slice(i));
  }

  CheckMatrices(lsh.ReferenceSet(), xmlLsh.ReferenceSet(),
      textLsh.ReferenceSet(), binaryLsh.ReferenceSet());
  CheckMatrices(lsh.Offsets(), xmlLsh.Offsets(), textLsh.Offsets(),
      binaryLsh.Offsets());
  CheckMatrices(lsh.SecondHashWeights(), xmlLsh.SecondHashWeights(),
      textLsh.SecondHashWeights(), binaryLsh.SecondHashWeights());

  BOOST_REQUIRE_EQUAL(lsh.BucketSize(), xmlLsh.BucketSize());
  BOOST_REQUIRE_EQUAL(lsh.BucketSize(), textLsh.BucketSize());
  BOOST_REQUIRE_EQUAL(lsh.BucketSize(), binaryLsh.BucketSize());

  BOOST_REQUIRE_EQUAL(lsh.SecondHashTable().size(),
      xmlLsh.SecondHashTable().size());
  BOOST_REQUIRE_EQUAL(lsh.SecondHashTable().size(),
      textLsh.SecondHashTable().size());
  BOOST_REQUIRE_EQUAL(lsh.SecondHashTable().size(),
      binaryLsh.SecondHashTable().size());

  for (size_t i = 0; i < lsh.SecondHashTable().size(); ++i)
  CheckMatrices(lsh.SecondHashTable()[i], xmlLsh.SecondHashTable()[i],
      textLsh.SecondHashTable()[i], binaryLsh.SecondHashTable()[i]);
}

// Make sure serialization works for the decision stump.
BOOST_AUTO_TEST_CASE(DecisionStumpTest)
{
  // Generate dataset.
  arma::mat trainingData = arma::randu<arma::mat>(4, 100);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < 25; ++i)
    labels[i] = 0;
  for (size_t i = 25; i < 50; ++i)
    labels[i] = 3;
  for (size_t i = 50; i < 75; ++i)
    labels[i] = 1;
  for (size_t i = 75; i < 100; ++i)
    labels[i] = 2;

  DecisionStump<> ds(trainingData, labels, 4, 3);

  arma::mat otherData = arma::randu<arma::mat>(3, 100);
  arma::Row<size_t> otherLabels = arma::randu<arma::Row<size_t>>(100);
  DecisionStump<> xmlDs(otherData, otherLabels, 2, 3);

  DecisionStump<> textDs;
  DecisionStump<> binaryDs(trainingData, labels, 4, 10);

  SerializeObjectAll(ds, xmlDs, textDs, binaryDs);

  // Make sure that everything is the same about the new decision stumps.
  BOOST_REQUIRE_EQUAL(ds.SplitDimension(), xmlDs.SplitDimension());
  BOOST_REQUIRE_EQUAL(ds.SplitDimension(), textDs.SplitDimension());
  BOOST_REQUIRE_EQUAL(ds.SplitDimension(), binaryDs.SplitDimension());

  CheckMatrices(ds.Split(), xmlDs.Split(), textDs.Split(), binaryDs.Split());
  CheckMatrices(ds.BinLabels(), xmlDs.BinLabels(), textDs.BinLabels(),
      binaryDs.BinLabels());
}

// Make sure serialization works for LARS.
BOOST_AUTO_TEST_CASE(LARSTest)
{
  using namespace mlpack::regression;

  // Create a dataset.
  arma::mat X = arma::randn(75, 250);
  arma::vec beta = arma::randn(75, 1);
  arma::vec y = trans(X) * beta;

  LARS lars(true, 0.1, 0.1);
  arma::vec betaOpt;
  lars.Train(X, y, betaOpt);

  // Now, serialize.
  LARS xmlLars(false, 0.5, 0.0), binaryLars(true, 1.0, 0.0),
      textLars(false, 0.1, 0.1);

  // Train textLars.
  arma::mat textX = arma::randn(25, 150);
  arma::vec textBeta = arma::randn(25, 1);
  arma::vec textY = trans(textX) * textBeta;
  arma::vec textBetaOpt;
  textLars.Train(textX, textY, textBetaOpt);

  SerializeObjectAll(lars, xmlLars, binaryLars, textLars);

  // Now, check that predictions are the same.
  arma::vec pred, xmlPred, textPred, binaryPred;
  lars.Predict(X, pred);
  xmlLars.Predict(X, xmlPred);
  textLars.Predict(X, textPred);
  binaryLars.Predict(X, binaryPred);

  CheckMatrices(pred, xmlPred, textPred, binaryPred);
}

/**
 * Test serialization of the HoeffdingNumericSplit object after binning has
 * occured.
 */
BOOST_AUTO_TEST_CASE(HoeffdingNumericSplitTest)
{
  using namespace mlpack::tree;

  HoeffdingNumericSplit<GiniImpurity> split(3);
  // Train until it bins.
  for (size_t i = 0; i < 200; ++i)
    split.Train(mlpack::math::Random(), mlpack::math::RandInt(3));

  HoeffdingNumericSplit<GiniImpurity> xmlSplit(5);
  HoeffdingNumericSplit<GiniImpurity> textSplit(7);
  for (size_t i = 0; i < 200; ++i)
    textSplit.Train(mlpack::math::Random() + 3, 0);
  HoeffdingNumericSplit<GiniImpurity> binarySplit(2);

  SerializeObjectAll(split, xmlSplit, textSplit, binarySplit);

  // Ensure that everything is the same.
  BOOST_REQUIRE_EQUAL(split.Bins(), xmlSplit.Bins());
  BOOST_REQUIRE_EQUAL(split.Bins(), textSplit.Bins());
  BOOST_REQUIRE_EQUAL(split.Bins(), binarySplit.Bins());

  double bestSplit, secondBestSplit;
  double baseBestSplit, baseSecondBestSplit;
  split.EvaluateFitnessFunction(baseBestSplit, baseSecondBestSplit);
  xmlSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  BOOST_REQUIRE_CLOSE(bestSplit, baseBestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-10);

  textSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  BOOST_REQUIRE_CLOSE(bestSplit, baseBestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-10);

  binarySplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  BOOST_REQUIRE_CLOSE(bestSplit, baseBestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-10);

  arma::Col<size_t> children, xmlChildren, textChildren, binaryChildren;
  NumericSplitInfo<double> splitInfo, xmlSplitInfo, textSplitInfo,
      binarySplitInfo;

  split.Split(children, splitInfo);
  xmlSplit.Split(xmlChildren, xmlSplitInfo);
  binarySplit.Split(binaryChildren, binarySplitInfo);
  textSplit.Split(textChildren, textSplitInfo);

  BOOST_REQUIRE_EQUAL(children.size(), xmlChildren.size());
  BOOST_REQUIRE_EQUAL(children.size(), textChildren.size());
  BOOST_REQUIRE_EQUAL(children.size(), binaryChildren.size());
  for (size_t i = 0; i < children.size(); ++i)
  {
    BOOST_REQUIRE_EQUAL(children[i], xmlChildren[i]);
    BOOST_REQUIRE_EQUAL(children[i], textChildren[i]);
    BOOST_REQUIRE_EQUAL(children[i], binaryChildren[i]);
  }

  // Random checks.
  for (size_t i = 0; i < 200; ++i)
  {
    const double random = mlpack::math::Random() * 1.5;
    BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(random),
                        xmlSplitInfo.CalculateDirection(random));
    BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(random),
                        textSplitInfo.CalculateDirection(random));
    BOOST_REQUIRE_EQUAL(splitInfo.CalculateDirection(random),
                        binarySplitInfo.CalculateDirection(random));
  }
}

/**
 * Make sure serialization of the HoeffdingNumericSplit object before binning
 * occurs is successful.
 */
BOOST_AUTO_TEST_CASE(HoeffdingNumericSplitBeforeBinningTest)
{
  using namespace mlpack::tree;

  HoeffdingNumericSplit<GiniImpurity> split(3);
  // Train but not until it bins.
  for (size_t i = 0; i < 50; ++i)
    split.Train(mlpack::math::Random(), mlpack::math::RandInt(3));

  HoeffdingNumericSplit<GiniImpurity> xmlSplit(5);
  HoeffdingNumericSplit<GiniImpurity> textSplit(7);
  for (size_t i = 0; i < 200; ++i)
    textSplit.Train(mlpack::math::Random() + 3, 0);
  HoeffdingNumericSplit<GiniImpurity> binarySplit(2);

  SerializeObjectAll(split, xmlSplit, textSplit, binarySplit);

  // Ensure that everything is the same.
  BOOST_REQUIRE_EQUAL(split.Bins(), xmlSplit.Bins());
  BOOST_REQUIRE_EQUAL(split.Bins(), textSplit.Bins());
  BOOST_REQUIRE_EQUAL(split.Bins(), binarySplit.Bins());

  double baseBestSplit, baseSecondBestSplit;
  double bestSplit, secondBestSplit;
  split.EvaluateFitnessFunction(baseBestSplit, baseSecondBestSplit);
  textSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);

  BOOST_REQUIRE_SMALL(baseBestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(baseSecondBestSplit, 1e-5);

  BOOST_REQUIRE_SMALL(bestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-5);

  xmlSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  BOOST_REQUIRE_SMALL(bestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-5);

  binarySplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  BOOST_REQUIRE_SMALL(bestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-5);
}

/**
 * Make sure the HoeffdingCategoricalSplit object serializes correctly.
 */
BOOST_AUTO_TEST_CASE(HoeffdingCategoricalSplitTest)
{
  using namespace mlpack::tree;

  HoeffdingCategoricalSplit<GiniImpurity> split(10, 3);
  for (size_t i = 0; i < 50; ++i)
    split.Train(mlpack::math::RandInt(10), mlpack::math::RandInt(3));

  HoeffdingCategoricalSplit<GiniImpurity> xmlSplit(3, 7);
  HoeffdingCategoricalSplit<GiniImpurity> binarySplit(4, 11);
  HoeffdingCategoricalSplit<GiniImpurity> textSplit(2, 2);
  for (size_t i = 0; i < 10; ++i)
    textSplit.Train(mlpack::math::RandInt(2), mlpack::math::RandInt(2));

  SerializeObjectAll(split, xmlSplit, textSplit, binarySplit);

  BOOST_REQUIRE_EQUAL(split.MajorityClass(), xmlSplit.MajorityClass());
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), textSplit.MajorityClass());
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), binarySplit.MajorityClass());

  double bestSplit, secondBestSplit;
  double baseBestSplit, baseSecondBestSplit;
  split.EvaluateFitnessFunction(baseBestSplit, baseSecondBestSplit);
  xmlSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);

  BOOST_REQUIRE_CLOSE(bestSplit, baseBestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-10);

  textSplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  BOOST_REQUIRE_CLOSE(bestSplit, baseBestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-10);

  binarySplit.EvaluateFitnessFunction(bestSplit, secondBestSplit);
  BOOST_REQUIRE_CLOSE(bestSplit, baseBestSplit, 1e-5);
  BOOST_REQUIRE_SMALL(secondBestSplit, 1e-10);

  arma::Col<size_t> children, xmlChildren, textChildren, binaryChildren;
  CategoricalSplitInfo splitInfo(1); // I don't care about this.

  split.Split(children, splitInfo);
  xmlSplit.Split(xmlChildren, splitInfo);
  binarySplit.Split(binaryChildren, splitInfo);
  textSplit.Split(textChildren, splitInfo);

  BOOST_REQUIRE_EQUAL(children.size(), xmlChildren.size());
  BOOST_REQUIRE_EQUAL(children.size(), textChildren.size());
  BOOST_REQUIRE_EQUAL(children.size(), binaryChildren.size());
  for (size_t i = 0; i < children.size(); ++i)
  {
    BOOST_REQUIRE_EQUAL(children[i], xmlChildren[i]);
    BOOST_REQUIRE_EQUAL(children[i], textChildren[i]);
    BOOST_REQUIRE_EQUAL(children[i], binaryChildren[i]);
  }
}

/**
 * Make sure the HoeffdingTree object serializes correctly before a split has
 * occured.
 */
BOOST_AUTO_TEST_CASE(HoeffdingTreeBeforeSplitTest)
{
  data::DatasetInfo info(5);
  info.MapString("0", 2); // Dimension 1 is categorical.
  info.MapString("1", 2);
  HoeffdingTree<> split(info, 2, 0.99, 15000, 1);

  // Train for 2 samples.
  split.Train(arma::vec("0.3 0.4 1 0.6 0.7"), 0);
  split.Train(arma::vec("-0.3 0.0 0 0.7 0.8"), 1);

  data::DatasetInfo wrongInfo(3);
  wrongInfo.MapString("1", 1);
  HoeffdingTree<> xmlSplit(wrongInfo, 7, 0.1, 10, 1);

  // Force the binarySplit to split.
  data::DatasetInfo binaryInfo(2);
  binaryInfo.MapString("cat0", 0);
  binaryInfo.MapString("cat1", 0);
  binaryInfo.MapString("cat0", 1);

  HoeffdingTree<> binarySplit(info, 2, 0.95, 5000, 1);

  // Feed samples from each class.
  for (size_t i = 0; i < 500; ++i)
  {
    binarySplit.Train(arma::Col<size_t>("0 0"), 0);
    binarySplit.Train(arma::Col<size_t>("1 0"), 1);
  }

  HoeffdingTree<> textSplit(wrongInfo, 11, 0.75, 1000, 1);

  SerializeObjectAll(split, xmlSplit, textSplit, binarySplit);

  BOOST_REQUIRE_EQUAL(split.SplitDimension(), xmlSplit.SplitDimension());
  BOOST_REQUIRE_EQUAL(split.SplitDimension(), binarySplit.SplitDimension());
  BOOST_REQUIRE_EQUAL(split.SplitDimension(), textSplit.SplitDimension());

  BOOST_REQUIRE_EQUAL(split.MajorityClass(), xmlSplit.MajorityClass());
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), binarySplit.MajorityClass());
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), textSplit.MajorityClass());

  BOOST_REQUIRE_EQUAL(split.SplitCheck(), xmlSplit.SplitCheck());
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), binarySplit.SplitCheck());
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), textSplit.SplitCheck());
}

/**
 * Make sure the HoeffdingTree object serializes correctly after a split has
 * occurred.
 */
BOOST_AUTO_TEST_CASE(HoeffdingTreeAfterSplitTest)
{
  // Force the split to split.
  data::DatasetInfo info(2);
  info.MapString("cat0", 0);
  info.MapString("cat1", 0);
  info.MapString("cat0", 1);

  HoeffdingTree<> split(info, 2, 0.95, 5000, 1);

  // Feed samples from each class.
  for (size_t i = 0; i < 500; ++i)
  {
    split.Train(arma::Col<size_t>("0 0"), 0);
    split.Train(arma::Col<size_t>("1 0"), 1);
  }
  // Ensure a split has happened.
  BOOST_REQUIRE_NE(split.SplitDimension(), size_t(-1));

  data::DatasetInfo wrongInfo(3);
  wrongInfo.MapString("1", 1);
  HoeffdingTree<> xmlSplit(wrongInfo, 7, 0.1, 10, 1);

  data::DatasetInfo binaryInfo(5);
  binaryInfo.MapString("0", 2); // Dimension 2 is categorical.
  binaryInfo.MapString("1", 2);
  HoeffdingTree<> binarySplit(binaryInfo, 2, 0.99, 15000, 1);

  // Train for 2 samples.
  binarySplit.Train(arma::vec("0.3 0.4 1 0.6 0.7"), 0);
  binarySplit.Train(arma::vec("-0.3 0.0 0 0.7 0.8"), 1);

  HoeffdingTree<> textSplit(wrongInfo, 11, 0.75, 1000, 1);

  SerializeObjectAll(split, xmlSplit, textSplit, binarySplit);

  BOOST_REQUIRE_EQUAL(split.SplitDimension(), xmlSplit.SplitDimension());
  BOOST_REQUIRE_EQUAL(split.SplitDimension(), binarySplit.SplitDimension());
  BOOST_REQUIRE_EQUAL(split.SplitDimension(), textSplit.SplitDimension());

  // If splitting has already happened, then SplitCheck() should return 0.
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), 0);
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), xmlSplit.SplitCheck());
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), binarySplit.SplitCheck());
  BOOST_REQUIRE_EQUAL(split.SplitCheck(), textSplit.SplitCheck());

  BOOST_REQUIRE_EQUAL(split.MajorityClass(), xmlSplit.MajorityClass());
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), binarySplit.MajorityClass());
  BOOST_REQUIRE_EQUAL(split.MajorityClass(), textSplit.MajorityClass());

  BOOST_REQUIRE_EQUAL(split.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")),
      xmlSplit.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")));
  BOOST_REQUIRE_EQUAL(split.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")),
      binarySplit.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")));
  BOOST_REQUIRE_EQUAL(split.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")),
      textSplit.CalculateDirection(arma::vec("0.3 0.4 1 0.6 0.7")));
}

BOOST_AUTO_TEST_CASE(EmptyHoeffdingTreeTest)
{
  using namespace mlpack::tree;

  data::DatasetInfo info(6);
  HoeffdingTree<> tree(info, 2);
  HoeffdingTree<> xmlTree(info, 3);
  HoeffdingTree<> binaryTree(info, 4);
  HoeffdingTree<> textTree(info, 5);

  SerializeObjectAll(tree, xmlTree, binaryTree, textTree);

  BOOST_REQUIRE_EQUAL(tree.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(xmlTree.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(binaryTree.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(textTree.NumChildren(), 0);
}

/**
 * Build a Hoeffding tree, then save it and make sure other trees can classify
 * as effectively.
 */
BOOST_AUTO_TEST_CASE(HoeffdingTreeTest)
{
  using namespace mlpack::tree;

  arma::mat dataset(2, 400);
  arma::Row<size_t> labels(400);
  for (size_t i = 0; i < 200; ++i)
  {
    dataset(0, 2 * i) = mlpack::math::RandInt(4);
    dataset(1, 2 * i) = mlpack::math::RandInt(2);
    dataset(0, 2 * i + 1) = mlpack::math::RandInt(4);
    dataset(1, 2 * i + 1) = mlpack::math::RandInt(2) + 2;
    labels[2 * i] = 0;
    labels[2 * i + 1] = 1;
  }
  // Make the features categorical.
  data::DatasetInfo info(2);
  info.MapString("a", 0);
  info.MapString("b", 0);
  info.MapString("c", 0);
  info.MapString("d", 0);
  info.MapString("a", 1);
  info.MapString("b", 1);
  info.MapString("c", 1);
  info.MapString("d", 1);

  HoeffdingTree<> tree(dataset, info, labels, 2, false /* no batch mode */);

  data::DatasetInfo xmlInfo(1);
  HoeffdingTree<> xmlTree(xmlInfo, 1);
  data::DatasetInfo binaryInfo(5);
  HoeffdingTree<> binaryTree(binaryInfo, 6);
  data::DatasetInfo textInfo(7);
  HoeffdingTree<> textTree(textInfo, 100);

  SerializeObjectAll(tree, xmlTree, textTree, binaryTree);

  BOOST_REQUIRE_EQUAL(tree.NumChildren(), xmlTree.NumChildren());
  BOOST_REQUIRE_EQUAL(tree.NumChildren(), textTree.NumChildren());
  BOOST_REQUIRE_EQUAL(tree.NumChildren(), binaryTree.NumChildren());

  BOOST_REQUIRE_EQUAL(tree.SplitDimension(), xmlTree.SplitDimension());
  BOOST_REQUIRE_EQUAL(tree.SplitDimension(), textTree.SplitDimension());
  BOOST_REQUIRE_EQUAL(tree.SplitDimension(), binaryTree.SplitDimension());

  for (size_t i = 0; i < tree.NumChildren(); ++i)
  {
    BOOST_REQUIRE_EQUAL(tree.Child(i).NumChildren(), 0);
    BOOST_REQUIRE_EQUAL(xmlTree.Child(i).NumChildren(), 0);
    BOOST_REQUIRE_EQUAL(binaryTree.Child(i).NumChildren(), 0);
    BOOST_REQUIRE_EQUAL(textTree.Child(i).NumChildren(), 0);

    BOOST_REQUIRE_EQUAL(tree.Child(i).SplitDimension(),
        xmlTree.Child(i).SplitDimension());
    BOOST_REQUIRE_EQUAL(tree.Child(i).SplitDimension(),
        textTree.Child(i).SplitDimension());
    BOOST_REQUIRE_EQUAL(tree.Child(i).SplitDimension(),
        binaryTree.Child(i).SplitDimension());
  }
}

BOOST_AUTO_TEST_SUITE_END();
