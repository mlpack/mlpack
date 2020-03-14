/**
 * @file kde_test.cpp
 * @author Roberto Hueso
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

// Brute force gaussian KDE.
template <typename KernelType>
void BruteForceKDE(const arma::mat& reference,
                   const arma::mat& query,
                   arma::vec& densities,
                   KernelType& kernel)
{
  metric::EuclideanDistance metric;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    for (size_t j = 0; j < reference.n_cols; ++j)
    {
      double distance = metric.Evaluate(query.col(i), reference.col(j));
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
  arma::vec estimationsResult = {0.08323668699564207296148765,
                                 0.00167470061366603324010116,
                                 0.07658867126520703394465527,
                                 0.01028120384800740999553525};
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree>
      kde(0.0, 0.01, GaussianKernel(0.8));
  kde.Train(reference);
  kde.Evaluate(query, estimations);
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(estimations[i], estimationsResult[i], 0.01);
}

/**
 * Test Train(Tree...) and Evaluate(Tree...).
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
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree>
      kde(0.0, 1e-6, GaussianKernel(kernelBandwidth));
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
  const double kernelBandwidth = 0.12;
  const double relError = 0.05;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
}

/**
 * Test single-tree implementation results against brute force results.
 */
BOOST_AUTO_TEST_CASE(GaussianSingleKDEBruteForceTest)
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.3;
  const double relError = 0.04;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
      kde(relError, 0.0, kernel, KDEMode::SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
}

/**
 * Test single-tree implementation results against brute force results using
 * a cover-tree and Epanechnikov kernel.
 */
BOOST_AUTO_TEST_CASE(EpanechnikovCoverSingleKDETest)
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 1.1;
  const double relError = 0.08;

  // Brute force KDE.
  EpanechnikovKernel kernel(kernelBandwidth);
  BruteForceKDE<EpanechnikovKernel>(reference,
                                    query,
                                    bfEstimations,
                                    kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<EpanechnikovKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
}

/**
 * Test single-tree implementation results against brute force results using
 * a cover-tree and Gaussian kernel.
 */
BOOST_AUTO_TEST_CASE(GaussianCoverSingleKDETest)
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 1.1;
  const double relError = 0.08;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
}

/**
 * Test single-tree implementation results against brute force results using
 * an octree and Epanechnikov kernel.
 */
BOOST_AUTO_TEST_CASE(EpanechnikovOctreeSingleKDETest)
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 1.0;
  const double relError = 0.05;

  // Brute force KDE.
  EpanechnikovKernel kernel(kernelBandwidth);
  BruteForceKDE<EpanechnikovKernel>(reference,
                                    query,
                                    bfEstimations,
                                    kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<EpanechnikovKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::Octree>
      kde(relError, 0.0, kernel, KDEMode::SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
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

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // BallTree KDE.
  typedef BallTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries, oldFromNewReferences;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 2);
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      BallTree>
      kde(relError, 0.0, GaussianKernel(kernelBandwidth));
  kde.Train(referenceTree, &oldFromNewReferences);
  kde.Evaluate(queryTree, std::move(oldFromNewQueries), treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);

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

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::Octree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
}

/**
 * Test RTree dual-tree implementation results against brute force results.
 */
BOOST_AUTO_TEST_CASE(RTreeGaussianKDETest)
{
  arma::mat reference = arma::randu(2, 500);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.3;
  const double relError = 0.01;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::RTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
}

/**
 * Test Standard Cover Tree dual-tree implementation results against brute
 * force results using Gaussian kernel.
 */
BOOST_AUTO_TEST_CASE(StandardCoverTreeGaussianKDETest)
{
  arma::mat reference = arma::randu(2, 500);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.3;
  const double relError = 0.01;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
}

/**
 * Test Standard Cover Tree dual-tree implementation results against brute
 * force results using Epanechnikov kernel.
 */
BOOST_AUTO_TEST_CASE(StandardCoverTreeEpanechnikovKDETest)
{
  arma::mat reference = arma::randu(2, 500);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.3;
  const double relError = 0.01;

  // Brute force KDE.
  EpanechnikovKernel kernel(kernelBandwidth);
  BruteForceKDE<EpanechnikovKernel>(reference,
                                    query,
                                    bfEstimations,
                                    kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<EpanechnikovKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
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

  // Duplicate value.
  reference.col(2) = reference.col(3);

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Dual-tree KDE.
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries, oldFromNewReferences;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 2);
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree>
      kde(relError, 0.0, GaussianKernel(kernelBandwidth));
  kde.Train(referenceTree, &oldFromNewReferences);
  kde.Evaluate(queryTree, oldFromNewQueries, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);

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

  // Duplicate value.
  query.col(2) = query.col(3);

  // Dual-tree KDE.
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries, oldFromNewReferences;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 2);
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree>
      kde(relError, 0.0, GaussianKernel(kernelBandwidth));
  kde.Train(referenceTree, &oldFromNewReferences);
  kde.Evaluate(queryTree, oldFromNewQueries, estimations);

  // Check whether results are equal.
  BOOST_REQUIRE_CLOSE(estimations[2], estimations[3], relError * 100);

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

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Breadth-First KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree,
      tree::KDTree<metric::EuclideanDistance,
                   kde::KDEStat,
                   arma::mat>::template BreadthFirstDualTreeTraverser>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
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

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(bfEstimations[i], treeEstimations[i], relError * 100);
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

  // KDE.
  metric::EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);

  // When training using the dataset matrix.
  BOOST_REQUIRE_THROW(kde.Train(reference), std::invalid_argument);

  // When training using a tree.
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

  // KDE.
  metric::EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);

  // When evaluating using the query dataset matrix.
  BOOST_REQUIRE_THROW(kde.Evaluate(query, estimations),
                    std::invalid_argument);

  // When evaluating using a query tree.
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
  // Set estimations to the wrong size.
  arma::vec estimations(33, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // KDE.
  metric::EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
      kde(relError, 0.0, kernel, KDEMode::DUAL_TREE_MODE, metric);
  kde.Train(reference);

  // The query set must be empty.
  BOOST_REQUIRE_EQUAL(query.n_cols, 0);
  // When evaluating using the query dataset matrix.
  BOOST_REQUIRE_NO_THROW(kde.Evaluate(query, estimations));

  // When evaluating using a query tree.
  typedef KDTree<EuclideanDistance, kde::KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 3);
  BOOST_REQUIRE_NO_THROW(
      kde.Evaluate(queryTree, oldFromNewQueries, estimations));
  delete queryTree;

  // Estimations must be empty.
  BOOST_REQUIRE_EQUAL(estimations.size(), 0);
}

/**
 * Tests serialiation of KDE models.
 */
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  // Initial KDE model to be serialized.
  const double relError = 0.25;
  const double absError = 0.0;
  const bool monteCarlo = false;
  const double MCProb = 0.8;
  const size_t initialSampleSize = 35;
  const double entryCoef = 5;
  const double breakCoef = 0.6;
  arma::mat reference = arma::randu(4, 800);
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
    kde(relError,
        absError,
        GaussianKernel(0.25),
        KDEMode::DUAL_TREE_MODE,
        metric::EuclideanDistance(),
        monteCarlo,
        MCProb,
        initialSampleSize,
        entryCoef,
        breakCoef);
  kde.Train(reference);

  // Get estimations to compare.
  arma::mat query = arma::randu(4, 100);;
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  kde.Evaluate(query, estimations);

  // Initialize serialized objects.
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
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

  const KDEMode mode = KDEMode::DUAL_TREE_MODE;
  BOOST_REQUIRE_EQUAL(kde.Mode(), mode);
  BOOST_REQUIRE_EQUAL(kdeXml.Mode(), mode);
  BOOST_REQUIRE_EQUAL(kdeText.Mode(), mode);
  BOOST_REQUIRE_EQUAL(kdeBinary.Mode(), mode);

  BOOST_REQUIRE_EQUAL(kde.MonteCarlo(), monteCarlo);
  BOOST_REQUIRE_EQUAL(kdeXml.MonteCarlo(), monteCarlo);
  BOOST_REQUIRE_EQUAL(kdeText.MonteCarlo(), monteCarlo);
  BOOST_REQUIRE_EQUAL(kdeBinary.MonteCarlo(), monteCarlo);

  BOOST_REQUIRE_CLOSE(kde.MCProb(), MCProb, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeXml.MCProb(), MCProb, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeText.MCProb(), MCProb, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeBinary.MCProb(), MCProb, 1e-8);

  BOOST_REQUIRE_EQUAL(kde.MCInitialSampleSize(), initialSampleSize);
  BOOST_REQUIRE_EQUAL(kdeXml.MCInitialSampleSize(), initialSampleSize);
  BOOST_REQUIRE_EQUAL(kdeText.MCInitialSampleSize(), initialSampleSize);
  BOOST_REQUIRE_EQUAL(kdeBinary.MCInitialSampleSize(), initialSampleSize);

  BOOST_REQUIRE_CLOSE(kde.MCEntryCoef(), entryCoef, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeXml.MCEntryCoef(), entryCoef, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeText.MCEntryCoef(), entryCoef, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeBinary.MCEntryCoef(), entryCoef, 1e-8);

  BOOST_REQUIRE_CLOSE(kde.MCBreakCoef(), breakCoef, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeXml.MCBreakCoef(), breakCoef, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeText.MCBreakCoef(), breakCoef, 1e-8);
  BOOST_REQUIRE_CLOSE(kdeBinary.MCBreakCoef(), breakCoef, 1e-8);

  // Test if execution gives the same result.
  arma::vec xmlEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec textEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec binEstimations = arma::vec(query.n_cols, arma::fill::zeros);

  kdeXml.Evaluate(query, xmlEstimations);
  kdeText.Evaluate(query, textEstimations);
  kdeBinary.Evaluate(query, binEstimations);

  for (size_t i = 0; i < query.n_cols; ++i)
  {
    BOOST_REQUIRE_CLOSE(estimations[i], xmlEstimations[i], relError * 100);
    BOOST_REQUIRE_CLOSE(estimations[i], textEstimations[i], relError * 100);
    BOOST_REQUIRE_CLOSE(estimations[i], binEstimations[i], relError * 100);
  }
}

/**
 * Test if the copy constructor and copy operator works properly.
 */
BOOST_AUTO_TEST_CASE(CopyConstructor)
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec estimations1, estimations2, estimations3;
  const double kernelBandwidth = 1.5;
  const double relError = 0.05;

  typedef KDE<GaussianKernel, metric::EuclideanDistance, arma::mat>
      KDEType;

  // KDE.
  KDEType kde(relError, 0, kernel::GaussianKernel(kernelBandwidth));
  kde.Train(std::move(reference));

  // Copy constructor KDE.
  KDEType constructor(kde);

  // Copy operator KDE.
  KDEType oper = kde;

  // Evaluations.
  kde.Evaluate(query, estimations1);
  constructor.Evaluate(query, estimations2);
  oper.Evaluate(query, estimations3);

  // Check results.
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    BOOST_REQUIRE_CLOSE(estimations1[i], estimations2[i], 1e-10);
    BOOST_REQUIRE_CLOSE(estimations2[i], estimations3[i], 1e-10);
  }
}

/**
 * Test if the move constructor works properly.
 */
BOOST_AUTO_TEST_CASE(MoveConstructor)
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec estimations1, estimations2, estimations3;
  const double kernelBandwidth = 1.2;
  const double relError = 0.05;

  typedef KDE<EpanechnikovKernel, metric::EuclideanDistance, arma::mat>
      KDEType;

  // KDE.
  KDEType kde(relError, 0, kernel::EpanechnikovKernel(kernelBandwidth));
  kde.Train(std::move(reference));
  kde.Evaluate(query, estimations1);

  // Move constructor KDE.
  KDEType constructor(std::move(kde));
  constructor.Evaluate(query, estimations2);

  // Check results.
  BOOST_REQUIRE_THROW(kde.Evaluate(query, estimations3), std::runtime_error);
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(estimations1[i], estimations2[i], 1e-10);
}

/**
 * Test if an untrained KDE works properly.
 */
BOOST_AUTO_TEST_CASE(NotTrained)
{
  arma::mat query = arma::randu(1, 10);
  std::vector<size_t> oldFromNew;
  arma::vec estimations;

  KDE<> kde;
  KDE<>::Tree queryTree(query, oldFromNew);

  // Check results.
  BOOST_REQUIRE_THROW(kde.Evaluate(query, estimations), std::runtime_error);
  BOOST_REQUIRE_THROW(kde.Evaluate(&queryTree, oldFromNew, estimations),
                      std::runtime_error);
  BOOST_REQUIRE_THROW(kde.Evaluate(estimations), std::runtime_error);
}

/**
 * Test single KD-tree implementation results against brute force results using
 * Monte Carlo estimations when possible.
 */
BOOST_AUTO_TEST_CASE(GaussianSingleKDTreeMonteCarloKDE)
{
  arma::mat reference = arma::randu(2, 3000);
  arma::mat query = arma::randu(2, 100);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.35;
  const double relError = 0.05;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
    kde(relError,
        0.0,
        kernel,
        KDEMode::SINGLE_TREE_MODE,
        metric,
        true,
        0.95,
        100,
        2,
        0.7);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // The Monte Carlo estimation has a random component so it can fail. Therefore
  // we require a reasonable amount of results to be right.
  size_t correctResults = 0;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    const double resultRelativeError =
      std::abs((bfEstimations[i] - treeEstimations[i]) / bfEstimations[i]);
    if (resultRelativeError < relError)
      ++correctResults;
  }

  BOOST_REQUIRE_GT(correctResults, 70);
}

/**
 * Test single cover-tree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
BOOST_AUTO_TEST_CASE(GaussianSingleCoverTreeMonteCarloKDE)
{
  arma::mat reference = arma::randu(2, 3000);
  arma::mat query = arma::randu(2, 100);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.35;
  const double relError = 0.05;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::StandardCoverTree>
    kde(relError,
        0.0,
        kernel,
        KDEMode::SINGLE_TREE_MODE,
        metric,
        true,
        0.95,
        100,
        2,
        0.7);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // The Monte Carlo estimation has a random component so it can fail. Therefore
  // we require a reasonable amount of results to be right.
  size_t correctResults = 0;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    const double resultRelativeError =
      std::abs((bfEstimations[i] - treeEstimations[i]) / bfEstimations[i]);
    if (resultRelativeError < relError)
      ++correctResults;
  }

  BOOST_REQUIRE_GT(correctResults, 70);
}

/**
 * Test single octree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
BOOST_AUTO_TEST_CASE(GaussianSingleOctreeMonteCarloKDE)
{
  arma::mat reference = arma::randu(2, 3000);
  arma::mat query = arma::randu(2, 100);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.55;
  const double relError = 0.02;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::Octree>
    kde(relError,
        0.0,
        kernel,
        KDEMode::SINGLE_TREE_MODE,
        metric,
        true,
        0.95,
        100,
        3,
        0.8);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // The Monte Carlo estimation has a random component so it can fail. Therefore
  // we require a reasonable amount of results to be right.
  size_t correctResults = 0;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    const double resultRelativeError =
      std::abs((bfEstimations[i] - treeEstimations[i]) / bfEstimations[i]);
    if (resultRelativeError < relError)
      ++correctResults;
  }

  BOOST_REQUIRE_GT(correctResults, 70);
}

/**
 * Test dual kd-tree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
BOOST_AUTO_TEST_CASE(GaussianDualKDTreeMonteCarloKDE)
{
  arma::mat reference = arma::randu(2, 3000);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.4;
  const double relError = 0.05;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree>
    kde(relError,
        0.0,
        kernel,
        KDEMode::DUAL_TREE_MODE,
        metric,
        true,
        0.95,
        100,
        3,
        0.8);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // The Monte Carlo estimation has a random component so it can fail. Therefore
  // we require a reasonable amount of results to be right.
  size_t correctResults = 0;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    const double resultRelativeError =
      std::abs((bfEstimations[i] - treeEstimations[i]) / bfEstimations[i]);
    if (resultRelativeError < relError)
      ++correctResults;
  }

  BOOST_REQUIRE_GT(correctResults, 70);
}

/**
 * Test dual Cover-tree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
BOOST_AUTO_TEST_CASE(GaussianDualCoverTreeMonteCarloKDE)
{
  arma::mat reference = arma::randu(2, 3000);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.5;
  const double relError = 0.025;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::StandardCoverTree>
    kde(relError,
        0.0,
        kernel,
        KDEMode::DUAL_TREE_MODE,
        metric,
        true,
        0.95,
        100,
        3,
        0.8);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // The Monte Carlo estimation has a random component so it can fail. Therefore
  // we require a reasonable amount of results to be right.
  size_t correctResults = 0;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    const double resultRelativeError =
      std::abs((bfEstimations[i] - treeEstimations[i]) / bfEstimations[i]);
    if (resultRelativeError < relError)
      ++correctResults;
  }

  BOOST_REQUIRE_GT(correctResults, 70);
}

/**
 * Test dual octree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
BOOST_AUTO_TEST_CASE(GaussianDualOctreeMonteCarloKDE)
{
  arma::mat reference = arma::randu(2, 3000);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.03;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::Octree>
    kde(relError,
        0.0,
        kernel,
        KDEMode::DUAL_TREE_MODE,
        metric,
        true,
        0.95,
        100,
        3,
        0.8);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // The Monte Carlo estimation has a random component so it can fail. Therefore
  // we require a reasonable amount of results to be right.
  size_t correctResults = 0;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    const double resultRelativeError =
      std::abs((bfEstimations[i] - treeEstimations[i]) / bfEstimations[i]);
    if (resultRelativeError < relError)
      ++correctResults;
  }

  BOOST_REQUIRE_GT(correctResults, 70);
}

/**
 * Test dual kd-tree breadth first traversal implementation results against
 * brute force results using Monte Carlo estimations when possible.
 */
BOOST_AUTO_TEST_CASE(GaussianBreadthDualKDTreeMonteCarloKDE)
{
  arma::mat reference = arma::randu(2, 3000);
  arma::mat query = arma::randu(2, 200);
  arma::vec bfEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec treeEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.025;

  // Brute force KDE.
  GaussianKernel kernel(kernelBandwidth);
  BruteForceKDE<GaussianKernel>(reference,
                                query,
                                bfEstimations,
                                kernel);

  // Optimized KDE.
  metric::EuclideanDistance metric;
  KDE<GaussianKernel,
      metric::EuclideanDistance,
      arma::mat,
      tree::KDTree,
      tree::KDTree<metric::EuclideanDistance,
                   kde::KDEStat,
                   arma::mat>::template BreadthFirstDualTreeTraverser>
    kde(relError,
        0.0,
        kernel,
        KDEMode::DUAL_TREE_MODE,
        metric,
        true,
        0.95,
        100,
        3,
        0.8);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // The Monte Carlo estimation has a random component so it can fail. Therefore
  // we require a reasonable amount of results to be right.
  size_t correctResults = 0;
  for (size_t i = 0; i < query.n_cols; ++i)
  {
    const double resultRelativeError =
      std::abs((bfEstimations[i] - treeEstimations[i]) / bfEstimations[i]);
    if (resultRelativeError < relError)
      ++correctResults;
  }

  BOOST_REQUIRE_GT(correctResults, 70);
}

BOOST_AUTO_TEST_SUITE_END();
