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
  ofstream ofs("test",ios::binary);
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
  ifstream ifs("test",ios::binary);
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
  //TestArmadilloSerialization<MatType, binary_iarchive, binary_oarchive>(x);
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
	ofstream ofs("test", ios::binary);
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

	ifstream ifs("test", ios::binary);
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
  ofstream ofs("test",ios::binary);
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

  ifstream ifs("test",ios::binary);
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

void CheckMatrices(const Mat<size_t>& x,
                   const Mat<size_t>& xmlX,
                   const Mat<size_t>& textX,
                   const Mat<size_t>& binaryX)
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
    BOOST_REQUIRE_EQUAL(x[i], xmlX[i]);
    BOOST_REQUIRE_EQUAL(x[i], textX[i]);
    BOOST_REQUIRE_EQUAL(x[i], binaryX[i]);
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

BOOST_AUTO_TEST_CASE(AllkNNTest)
{
  using neighbor::AllkNN;
  arma::mat dataset = arma::randu<arma::mat>(5, 2000);

  AllkNN allknn(dataset, false, false);

  AllkNN knnXml, knnText, knnBinary;

  SerializeObjectAll(allknn, knnXml, knnText, knnBinary);

  // Now run nearest neighbor and make sure the results are the same.
  arma::mat querySet = arma::randu<arma::mat>(5, 1000);

  arma::mat distances, xmlDistances, textDistances, binaryDistances;
  arma::Mat<size_t> neighbors, xmlNeighbors, textNeighbors, binaryNeighbors;

  allknn.Search(querySet, 5, neighbors, distances);
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


template<class T, class U>
void TestStack(T& stack, U& Stack)
	{
	using det::DTree;
	while (!stack.empty())
		{
		// Get the top node from the stack.
		DTree* node = stack.top();
		DTree* Node = Stack.top();

		stack.pop();
		Stack.pop();

		// Check that all the members are the same.
		BOOST_REQUIRE_EQUAL(node->Start(), Node->Start());

		BOOST_REQUIRE_EQUAL(node->End(), Node->End());

		BOOST_REQUIRE_EQUAL(node->SplitDim(), Node->SplitDim());

		if (std::abs(node->SplitValue()) < 1e-5)
			{
			BOOST_REQUIRE_SMALL(Node->SplitValue(), 1e-5);
			}
		else
			{
			BOOST_REQUIRE_CLOSE(node->SplitValue(), Node->SplitValue(), 1e-5);
			}

		if (std::abs(node->LogNegError()) < 1e-5)
			{
			BOOST_REQUIRE_SMALL(Node->LogNegError(), 1e-5);
			}
		else
			{
			BOOST_REQUIRE_CLOSE(node->LogNegError(), Node->LogNegError(), 1e-5);
			}

		if (std::abs(node->SubtreeLeavesLogNegError()) < 1e-5)
			{
			BOOST_REQUIRE_SMALL(Node->SubtreeLeavesLogNegError(), 1e-5);
			}
		else
			{
			BOOST_REQUIRE_CLOSE(node->SubtreeLeavesLogNegError(),
				Node->SubtreeLeavesLogNegError(), 1e-5);
			}

		BOOST_REQUIRE_EQUAL(node->SubtreeLeaves(), Node->SubtreeLeaves());

		if (std::abs(node->Ratio()) < 1e-5)
			{
			BOOST_REQUIRE_SMALL(Node->Ratio(), 1e-5);
			}
		else
			{
			BOOST_REQUIRE_CLOSE(node->Ratio(), Node->Ratio(), 1e-5);
			}

		if (std::abs(node->LogVolume()) < 1e-5)
			{
			BOOST_REQUIRE_SMALL(Node->LogVolume(), 1e-5);
			}
		else
			{
			BOOST_REQUIRE_CLOSE(node->LogVolume(), Node->LogVolume(), 1e-5);
			}

		if (node->Left() == NULL)
			{
			BOOST_REQUIRE(Node->Left() == NULL);
			}
		else
			{
			BOOST_REQUIRE(Node->Left() != NULL);

			// Push children onto stack.
			stack.push(node->Left());
			Stack.push(Node->Left());
			}

		if (node->Right() == NULL)
			{
			BOOST_REQUIRE(Node->Right() == NULL);
			}
		else
			{
			BOOST_REQUIRE(Node->Right() != NULL);

			// Push children onto stack.
			stack.push(node->Right());
			Stack.push(Node->Right());
			}

		BOOST_REQUIRE_EQUAL(node->Root(), Node->Root());

		if (std::abs(node->AlphaUpper()) < 1e-5)
			{
			BOOST_REQUIRE_SMALL(Node->AlphaUpper(), 1e-5);
			}
		else
			{
			BOOST_REQUIRE_CLOSE(node->AlphaUpper(), Node->AlphaUpper(), 1e-5);
			}

		BOOST_REQUIRE_EQUAL(node->MaxVals().n_elem, Node->MaxVals().n_elem);
		for (size_t i = 0; i < node->MaxVals().n_elem; ++i)
			{
			if (std::abs(node->MaxVals()[i]) < 1e-5)
				{
				BOOST_REQUIRE_SMALL(Node->MaxVals()[i], 1e-5);
				}
			else
				{
				BOOST_REQUIRE_CLOSE(node->MaxVals()[i], Node->MaxVals()[i], 1e-5);
				}
			}

		BOOST_REQUIRE_EQUAL(node->MinVals().n_elem, Node->MinVals().n_elem);
		for (size_t i = 0; i < node->MinVals().n_elem; ++i)
			{
			if (std::abs(node->MinVals()[i]) < 1e-5)
				{
				BOOST_REQUIRE_SMALL(Node->MinVals()[i], 1e-5);
				}
			else
				{
				BOOST_REQUIRE_CLOSE(node->MinVals()[i], Node->MinVals()[i], 1e-5);
				}
			}
		}

	}
BOOST_AUTO_TEST_CASE(DETXmlTest)
	{
	using det::DTree;

	// Create a density estimation tree on a random dataset.
	arma::mat dataset = arma::randu<arma::mat>(25, 5000);

	DTree tree(dataset);

	arma::mat otherDataset = arma::randu<arma::mat>(5, 100);
	DTree Tree;

	SerializeObject<DTree, xml_iarchive,xml_oarchive>(tree, Tree);

	std::stack<DTree*> stack, Stack;
	stack.push(&tree);
	Stack.push(&Tree);

	TestStack(stack, Stack);
		
	}


	BOOST_AUTO_TEST_CASE(DETTextTest)
		{
		using det::DTree;

		// Create a density estimation tree on a random dataset.
		arma::mat dataset = arma::randu<arma::mat>(25, 5000);

		DTree tree(dataset);

		arma::mat otherDataset = arma::randu<arma::mat>(5, 100);
		DTree Tree;

		SerializeObject<DTree, text_iarchive, text_oarchive>(tree, Tree);

		std::stack<DTree*> stack, Stack;
		stack.push(&tree);
		Stack.push(&Tree);

		TestStack(stack, Stack);

		}
		BOOST_AUTO_TEST_CASE(DETBInaryTest)
			{
			using det::DTree;

			// Create a density estimation tree on a random dataset.
			arma::mat dataset = arma::randu<arma::mat>(25, 5000);

			DTree tree(dataset);

			arma::mat otherDataset = arma::randu<arma::mat>(5, 100);
			DTree Tree;

			SerializeObject<DTree, binary_iarchive, binary_oarchive>(tree, Tree);

			std::stack<DTree*> stack, Stack;
			stack.push(&tree);
			Stack.push(&Tree);
			TestStack(stack, Stack);

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
  using neighbor::AllkNN;
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

  AllkNN allknn(dataset); // Exact search.
  allknn.Search(querySet, 10, neighbors, distances);
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
    CheckMatrices(lsh.Projection(i), xmlLsh.Projection(i),
        textLsh.Projection(i), binaryLsh.Projection(i));
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

  CheckMatrices(lsh.SecondHashTable(), xmlLsh.SecondHashTable(),
      textLsh.SecondHashTable(), binaryLsh.SecondHashTable());
}

BOOST_AUTO_TEST_SUITE_END();
