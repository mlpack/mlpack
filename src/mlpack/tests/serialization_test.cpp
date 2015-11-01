/**
 * @file serialization_test.cpp
 * @author Ryan Curtin
 *
 * Test serialization of mlpack objects.
 */
#include <boost/serialization/serialization.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#include <mlpack/core/dists/regression_distribution.hpp>
#include <mlpack/core/tree/ballbound.hpp>
#include <mlpack/core/tree/hrectbound.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>

using namespace mlpack;
using namespace mlpack::distribution;
using namespace mlpack::regression;
using namespace mlpack::bound;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace arma;
using namespace boost;
using namespace boost::archive;
using namespace boost::serialization;
using namespace std;

BOOST_AUTO_TEST_SUITE(SerializationTest);

// Test function for loading and saving Armadillo objects.
template<typename MatType,
         typename IArchiveType,
         typename OArchiveType>
void TestArmadilloSerialization(MatType& x)
{
  // First save it.
  ofstream ofs("test");
  OArchiveType o(ofs);

  bool success = true;
  try
  {
    o << BOOST_SERIALIZATION_NVP(x);
  }
  catch (archive_exception& e)
  {
    std::cerr << e.what();
    success = false;
  }

  BOOST_REQUIRE_EQUAL(success, true);
  ofs.close();

  // Now load it.
  MatType orig(x);
  success = true;
  ifstream ifs("test");
  IArchiveType i(ifs);

  try
  {
    i >> BOOST_SERIALIZATION_NVP(x);
  }
  catch (archive_exception& e)
  {
    std::cerr << e.what();
    success = false;
  }

  BOOST_REQUIRE_EQUAL(success, true);

  BOOST_REQUIRE_EQUAL(x.n_rows, orig.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_cols, orig.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_elem, orig.n_elem);

  for (size_t i = 0; i < x.n_cols; ++i)
    for (size_t j = 0; j < x.n_rows; ++j)
      if (double(orig(j, i)) == 0.0)
        BOOST_REQUIRE_SMALL(double(x(j, i)), 1e-8);
      else
        BOOST_REQUIRE_CLOSE(double(orig(j, i)), double(x(j, i)), 1e-8);

  remove("test");
}

// Test all serialization strategies.
template<typename MatType>
void TestAllArmadilloSerialization(MatType& x)
{
  TestArmadilloSerialization<MatType, xml_iarchive, xml_oarchive>(x);
  TestArmadilloSerialization<MatType, text_iarchive, text_oarchive>(x);
  TestArmadilloSerialization<MatType, binary_iarchive, binary_oarchive>(x);
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
BOOST_AUTO_TEST_CASE(ColSerializeXMLTest)
{
  arma::vec m;
  m.randu(50, 1);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
BOOST_AUTO_TEST_CASE(RowSerializeXMLTest)
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
BOOST_AUTO_TEST_CASE(SparseMatrixSerializeXMLTest)
{
  arma::sp_mat m;
  m.sprandu(50, 50, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about columns?
 */
BOOST_AUTO_TEST_CASE(SparseColSerializeXMLTest)
{
  arma::sp_vec m;
  m.sprandu(50, 1, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
BOOST_AUTO_TEST_CASE(SparseRowSerializeXMLTest)
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

// Save and load an mlpack object.
// The re-loaded copy is placed in 'newT'.
template<typename T, typename IArchiveType, typename OArchiveType>
void SerializeObject(T& t, T& newT)
{
  ofstream ofs("test");
  OArchiveType o(ofs);

  bool success = true;
  try
  {
    o << data::CreateNVP(t, "t");
  }
  catch (archive_exception& e)
  {
    std::cerr << e.what();
    success = false;
  }
  ofs.close();

  BOOST_REQUIRE_EQUAL(success, true);

  ifstream ifs("test");
  IArchiveType i(ifs);

  try
  {
    i >> data::CreateNVP(newT, "t");
  }
  catch (archive_exception& e)
  {
    std::cerr << e.what();
    success = false;
  }
  ifs.close();

  BOOST_REQUIRE_EQUAL(success, true);
}

// Test mlpack serialization with all three archive types.
template<typename T>
void SerializeObjectAll(T& t, T& xmlT, T& textT, T& binaryT)
{
  SerializeObject<T, text_iarchive, text_oarchive>(t, textT);
  SerializeObject<T, binary_iarchive, binary_oarchive>(t, binaryT);
  SerializeObject<T, xml_iarchive, xml_oarchive>(t, xmlT);
}

// Save and load a non-default-constructible mlpack object.
template<typename T, typename IArchiveType, typename OArchiveType>
void SerializePointerObject(T* t, T*& newT)
{
  ofstream ofs("test");
  OArchiveType o(ofs);

  bool success = true;
  try
  {
    o << data::CreateNVP(*t, "t");
  }
  catch (archive_exception& e)
  {
    std::cerr << e.what();
    success = false;
  }
  ofs.close();

  BOOST_REQUIRE_EQUAL(success, true);

  ifstream ifs("test");
  IArchiveType i(ifs);

  try
  {
    newT = new T(i);
  }
  catch (std::exception& e)
  {
    std::cerr << e.what();
    success = false;
  }
  ifs.close();

  BOOST_REQUIRE_EQUAL(success, true);
}

template<typename T>
void SerializePointerObjectAll(T* t, T*& xmlT, T*& textT, T*& binaryT)
{
  SerializePointerObject<T, text_iarchive, text_oarchive>(t, textT);
  SerializePointerObject<T, binary_iarchive, binary_oarchive>(t, binaryT);
  SerializePointerObject<T, xml_iarchive, xml_oarchive>(t, xmlT);
}

// Utility function to check the equality of two Armadillo matrices.
void CheckMatrices(const mat& x,
                   const mat& xmlX,
                   const mat& textX,
                   const mat& binaryX)
{
  // First check dimensions.
  BOOST_REQUIRE_EQUAL(x.n_rows, xmlX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, textX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, binaryX.n_rows);

  BOOST_REQUIRE_EQUAL(x.n_cols, xmlX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, textX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, binaryX.n_cols);

  BOOST_REQUIRE_EQUAL(x.n_elem, xmlX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, textX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, binaryX.n_elem);

  // Now check elements.
  for (size_t i = 0; i < x.n_elem; ++i)
  {
    const double val = x[i];
    if (val == 0.0)
    {
      BOOST_REQUIRE_SMALL(xmlX[i], 1e-8);
      BOOST_REQUIRE_SMALL(textX[i], 1e-8);
      BOOST_REQUIRE_SMALL(binaryX[i], 1e-8);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(val, xmlX[i], 1e-8);
      BOOST_REQUIRE_CLOSE(val, textX[i], 1e-8);
      BOOST_REQUIRE_CLOSE(val, binaryX[i], 1e-8);
    }
  }
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
  BallBound<arma::vec, MahalanobisDistance<>> b(100);
  b.Center().randu();
  b.Radius() = 14.0;
  b.Metric().Covariance().randu(100, 100);

  BallBound<arma::vec, MahalanobisDistance<>> xmlB, textB, binaryB;

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

  TreeType xmlTree(tree);
  TreeType textTree(tree);
  TreeType binaryTree(tree);

  SerializeObjectAll(tree, xmlTree, textTree, binaryTree);

  CheckTrees(tree, xmlTree, textTree, binaryTree);
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

  BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(),
      xmlSplit.EvaluateFitnessFunction(), 1e-5);
  BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(),
      textSplit.EvaluateFitnessFunction(), 1e-5);
  BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(),
      binarySplit.EvaluateFitnessFunction(), 1e-5);

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

  BOOST_REQUIRE_SMALL(split.EvaluateFitnessFunction(), 1e-5);
  BOOST_REQUIRE_SMALL(textSplit.EvaluateFitnessFunction(), 1e-5);
  BOOST_REQUIRE_SMALL(xmlSplit.EvaluateFitnessFunction(), 1e-5);
  BOOST_REQUIRE_SMALL(binarySplit.EvaluateFitnessFunction(), 1e-5);
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

  BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(),
                      xmlSplit.EvaluateFitnessFunction(), 1e-5);
  BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(),
                      textSplit.EvaluateFitnessFunction(), 1e-5);
  BOOST_REQUIRE_CLOSE(split.EvaluateFitnessFunction(),
                      binarySplit.EvaluateFitnessFunction(), 1e-5);

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

    BOOST_REQUIRE_EQUAL(tree.Child(i).MajorityClass(),
        xmlTree.Child(i).MajorityClass());
    BOOST_REQUIRE_EQUAL(tree.Child(i).MajorityClass(),
        textTree.Child(i).MajorityClass());
    BOOST_REQUIRE_EQUAL(tree.Child(i).MajorityClass(),
        binaryTree.Child(i).MajorityClass());
  }

  // Check that predictions are the same.
  arma::Row<size_t> predictions, xmlPredictions, binaryPredictions,
      textPredictions;
  tree.Classify(dataset, predictions);
  xmlTree.Classify(dataset, xmlPredictions);
  binaryTree.Classify(dataset, binaryPredictions);
  textTree.Classify(dataset, textPredictions);

  BOOST_REQUIRE_EQUAL(predictions.n_elem, xmlPredictions.n_elem);
  BOOST_REQUIRE_EQUAL(predictions.n_elem, textPredictions.n_elem);
  BOOST_REQUIRE_EQUAL(predictions.n_elem, binaryPredictions.n_elem);

  for (size_t i = 0; i < predictions.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(predictions[i], xmlPredictions[i]);
    BOOST_REQUIRE_EQUAL(predictions[i], textPredictions[i]);
    BOOST_REQUIRE_EQUAL(predictions[i], binaryPredictions[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END();
