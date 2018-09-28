/**
 * @file kde_test.cpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;

using namespace boost::serialization;

BOOST_AUTO_TEST_SUITE(KDETest);

// Brute force gaussian KDE
template <typename T>
void BruteForceKDE(const arma::mat& reference,
                   const arma::mat& query,
                   arma::vec& densities,
                   T& kernel)
{
  metric::EuclideanDistance metric;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    for (size_t j = 0; j < reference.n_cols; ++j)
    {
      double distance = metric.Evaluate(query.col(i),reference.col(j));
      densities(i) += kernel.Evaluate(distance);
    }
  }
  densities /= reference.n_cols;
}

/**
 * Test if simple case is correct according to manually calculated results.
 */
BOOST_AUTO_TEST_CASE(KDESimpleTest)
{
  // Transposed reference and query sets because it's easier to read.
  arma::mat reference = { {-1.0, -1.0},
                          {-2.0, -1.0},
                          {-3.0, -2.0},
                          { 1.0,  1.0},
                          { 2.0,  1.0},
                          { 3.0,  2.0} };
  arma::mat query = { { 0.0,  0.5},
                      { 0.4, -3.0},
                      { 0.0,  0.0},
                      {-2.1,  1.0} };
  arma::inplace_trans(reference);
  arma::inplace_trans(query);
  arma::vec estimations;
  // Manually calculated results.
  arma::vec estimations_result = {0.08323668699564207296148765,
                                  0.00167470061366603324010116,
                                  0.07658867126520703394465527,
                                  0.01028120384800740999553525};
  KDE<EuclideanDistance,
      arma::mat,
      GaussianKernel,
      KDTree>
    kde(0.8, 0.0, 0.01);
  kde.Train(reference);
  kde.Evaluate(query, estimations);
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(estimations[i], estimations_result[i], 0.01);
}

/**
 * Test Train(Tree...) and Evaluate(Tree...)
 */
BOOST_AUTO_TEST_CASE(KDETreeAsArguments)
{
  // Transposed reference and query sets because it's easier to read.
  arma::mat reference = { {-1.0, -1.0},
                          {-2.0, -1.0},
                          {-3.0, -2.0},
                          { 1.0,  1.0},
                          { 2.0,  1.0},
                          { 3.0,  2.0} };
  arma::mat query = { { 0.0,  0.5},
                      { 0.4, -3.0},
                      { 0.0,  0.0},
                      {-2.1,  1.0} };
  arma::inplace_trans(reference);
  arma::inplace_trans(query);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec estimationsResult = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.8;

  // Get brute force results.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                estimationsResult,
                                kernel);

  // Get dual-tree results.
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries, oldFromNewReferences;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 2);
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  KDE<EuclideanDistance,
      arma::mat,
      GaussianKernel,
      KDTree>
  kde(kernelBandwidth, 0.0, 1e-6);
  kde.Train(referenceTree, &oldFromNewReferences);
  kde.Evaluate(queryTree, std::move(oldFromNewQueries), estimations);
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(estimations[i], estimationsResult[i], 0.01);
  delete queryTree;
  delete referenceTree;
}

/**
 * Test dual-tree implementation results against brute force results.
 */
BOOST_AUTO_TEST_CASE(GaussianKDEBruteForceTest)
{
  arma::mat reference = arma::randu(2, 200);
  arma::mat query = arma::randu(2, 60);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.3;
  const double relError = 0.01;

  // Brute force KDE
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE
  metric::EuclideanDistance metric;
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree>
    kde(metric, kernel, relError, 0.0);
  kde.Train(reference);
  kde.Evaluate(std::move(query), treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError);
}

/**
 * Test BallTree dual-tree implementation results against brute force results.
 */
BOOST_AUTO_TEST_CASE(BallTreeGaussianKDETest)
{
  arma::mat reference = arma::randu(2, 200);
  arma::mat query = arma::randu(2, 60);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.4;
  const double relError = 0.05;

  // Brute force KDE
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // BallTree KDE
  typedef BallTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries, oldFromNewReferences;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 2);
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  KDE<EuclideanDistance,
      arma::mat,
      GaussianKernel,
      BallTree>
  kde(kernelBandwidth, relError, 0.0);
  kde.Train(referenceTree, &oldFromNewReferences);
  kde.Evaluate(queryTree, std::move(oldFromNewQueries), treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError);

  delete queryTree;
  delete referenceTree;
}

/**
 * Test Octree dual-tree implementation results against brute force results.
 */
BOOST_AUTO_TEST_CASE(OctreeGaussianKDETest)
{
  arma::mat reference = arma::randu(2, 500);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.3;
  const double relError = 0.01;

  // Brute force KDE
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE
  metric::EuclideanDistance metric;
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::Octree>
    kde(metric, kernel, relError, 0.0);
  kde.Train(reference);
  kde.Evaluate(std::move(query), treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError);
}

/**
 * Test duplicated value in reference matrix.
 */
BOOST_AUTO_TEST_CASE(DuplicatedReferenceSampleKDETest)
{
  arma::mat reference = arma::randu(2, 30);
  arma::mat query = arma::randu(2, 10);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.4;
  const double relError = 0.05;

  // Duplicate value
  reference.col(2) = reference.col(3);

  // Brute force KDE
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Dual-tree KDE
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries, oldFromNewReferences;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 2);
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  KDE<EuclideanDistance,
      arma::mat,
      GaussianKernel,
      KDTree>
  kde(kernelBandwidth, relError, 0.0);
  kde.Train(referenceTree, &oldFromNewReferences);
  kde.Evaluate(queryTree, oldFromNewQueries, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError);

  delete queryTree;
  delete referenceTree;
}

/**
 * Test duplicated value in query matrix.
 */
BOOST_AUTO_TEST_CASE(DuplicatedQuerySampleKDETest)
{
  arma::mat reference = arma::randu(2, 30);
  arma::mat query = arma::randu(2, 10);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.4;
  const double relError = 0.05;

  // Duplicate value
  query.col(2) = query.col(3);

  // Dual-tree KDE
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries, oldFromNewReferences;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 2);
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  KDE<EuclideanDistance,
      arma::mat,
      GaussianKernel,
      KDTree>
  kde(kernelBandwidth, relError, 0.0);
  kde.Train(referenceTree, &oldFromNewReferences);
  kde.Evaluate(queryTree, oldFromNewQueries, estimations);

  // Check whether results are equal.
  BOOST_REQUIRE_CLOSE(estimations[2], estimations[3], relError);

  delete queryTree;
  delete referenceTree;
}

/**
 * Test dual-tree breadth-first implementation results against brute force
 * results.
 */
BOOST_AUTO_TEST_CASE(BreadthFirstKDETest)
{
  arma::mat reference = arma::randu(2, 200);
  arma::mat query = arma::randu(2, 60);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.8;
  const double relError = 0.01;

  // Brute force KDE
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Breadth-First KDE
  metric::EuclideanDistance metric;
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree,
      tree::KDTree<metric::EuclideanDistance,
                   kde::KDEStat,
                   arma::mat>::template BreadthFirstDualTreeTraverser>
    kde(metric, kernel, relError, 0.0);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError);
}

/**
 * Test 1-dimensional implementation results against brute force results.
 */
BOOST_AUTO_TEST_CASE(OneDimensionalTest)
{
  arma::mat reference = arma::randu(1, 200);
  arma::mat query = arma::randu(1, 60);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // Brute force KDE
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE
  metric::EuclideanDistance metric;
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree>
    kde(metric, kernel, relError, 0.0);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError);
}

/**
 * Test a case where an empty reference set is given to train the model.
 */
BOOST_AUTO_TEST_CASE(EmptyReferenceTest)
{
  arma::mat reference;
  arma::mat query = arma::randu(1, 10);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // KDE
  metric::EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree>
    kde(metric, kernel, relError, 0.0);

  // When training using the dataset matrix
  BOOST_REQUIRE_THROW(kde.Train(reference), std::invalid_argument);

  // When training using a tree
  std::vector<size_t> oldFromNewReferences;
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  BOOST_REQUIRE_THROW(
      kde.Train(referenceTree, &oldFromNewReferences), std::invalid_argument);

  delete referenceTree;
}

/**
 * Tests when reference set values and query set values dimensions don't match.
 */
BOOST_AUTO_TEST_CASE(EvaluationMatchDimensionsTest)
{
  arma::mat reference = arma::randu(3, 10);
  arma::mat query = arma::randu(1, 10);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // KDE
  metric::EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree>
    kde(metric, kernel, relError, 0.0);
  kde.Train(reference);

  // When evaluating using the query dataset matrix
  BOOST_REQUIRE_THROW(kde.Evaluate(query, estimations),
                    std::invalid_argument);

  // When evaluating using a query tree
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 3);
  BOOST_REQUIRE_THROW(kde.Evaluate(queryTree, oldFromNewQueries, estimations),
                    std::invalid_argument);
  delete queryTree;
}

/**
 * Tests when an empty query set is given to be evaluated.
 */
BOOST_AUTO_TEST_CASE(EmptyQuerySetTest)
{
  arma::mat reference = arma::randu(1, 10);
  arma::mat query;
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // KDE
  metric::EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree>
    kde(metric, kernel, relError, 0.0);
  kde.Train(reference);

  // When evaluating using the query dataset matrix
  BOOST_REQUIRE_NO_THROW(kde.Evaluate(query, estimations));

  // When evaluating using a query tree
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 3);
  BOOST_REQUIRE_NO_THROW(
    kde.Evaluate(queryTree, oldFromNewQueries, estimations));

  delete queryTree;
}

/**
 * Tests serialiation of KDE models.
 */
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  // Initial KDE model to me serialized.
  const double relError = 0.25;
  const double absError = 0.0;
  arma::mat reference = arma::randu(4, 800);
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree>
    kde(0.25, relError, absError);
  kde.Train(reference);

  // Get estimations to compare.
  arma::mat query = arma::randu(4, 100);;
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  kde.Evaluate(query, estimations);

  // Initialize serialized objects.
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::KDTree> kdeXml, kdeText, kdeBinary;
  SerializeObjectAll(kde, kdeXml, kdeText, kdeBinary);

  // Check everything is correct.
  BOOST_REQUIRE_CLOSE(kde.RelativeError(), relError, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeXml.RelativeError(), relError, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeText.RelativeError(), relError, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeBinary.RelativeError(), relError, 1e-8);

  BOOST_REQUIRE_CLOSE(kde.AbsoluteError(), absError, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeXml.AbsoluteError(), absError, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeText.AbsoluteError(), absError, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeBinary.AbsoluteError(), absError, 1e-8);

  BOOST_REQUIRE_EQUAL(kde.IsTrained(), true);
  BOOST_REQUIRE_EQUAL(kdeXml.IsTrained(), true);
  BOOST_REQUIRE_EQUAL(kdeText.IsTrained(), true);
  BOOST_REQUIRE_EQUAL(kdeBinary.IsTrained(), true);

  // Test if execution gives the same result.
  arma::vec xmlEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec textEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec binEstimations = arma::vec(query.n_cols, arma::fill::zeros);

  kde.Evaluate(query, xmlEstimations);
  kde.Evaluate(query, textEstimations);
  kde.Evaluate(query, binEstimations);

  for (size_t i = 0; i < query.n_cols; ++i)
  {
    BOOST_REQUIRE_CLOSE(estimations[i], xmlEstimations[i], relError);
    BOOST_REQUIRE_CLOSE(estimations[i], textEstimations[i], relError);
    BOOST_REQUIRE_CLOSE(estimations[i], binEstimations[i], relError);
  }
}

BOOST_AUTO_TEST_SUITE_END();
