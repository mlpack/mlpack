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

BOOST_AUTO_TEST_SUITE_END();
