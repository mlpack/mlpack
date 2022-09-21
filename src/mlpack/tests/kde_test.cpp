/**
 * @file tests/kde_test.cpp
 * @author Roberto Hueso
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/kde.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace cereal;

// Brute force gaussian KDE.
template <typename KernelType>
void BruteForceKDE(const arma::mat& reference,
                   const arma::mat& query,
                   arma::vec& densities,
                   KernelType& kernel)
{
  EuclideanDistance metric;
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
TEST_CASE("KDESimpleTest", "[KDETest]")
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
    REQUIRE(estimations[i] == Approx(estimationsResult[i]).epsilon(0.001));
}

/**
 * Test Train(Tree...) and Evaluate(Tree...).
 */
TEST_CASE("KDETreeAsArguments", "[KDETest]")
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
  typedef KDTree<EuclideanDistance, KDEStat, arma::mat> Tree;
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
    REQUIRE(estimations[i] == Approx(estimationsResult[i]).epsilon(0.001));
  delete queryTree;
  delete referenceTree;
}

/**
 * Test dual-tree implementation results against brute force results.
 */
TEST_CASE("GaussianKDEBruteForceTest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kde(
      relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test single-tree implementation results against brute force results.
 */
TEST_CASE("GaussianSingleKDEBruteForceTest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kde(
      relError, 0.0, kernel, KDEMode::KDE_SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test single-tree implementation results against brute force results using
 * a cover-tree and Epanechnikov kernel.
 */
TEST_CASE("EpanechnikovCoverSingleKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<EpanechnikovKernel, EuclideanDistance, arma::mat, StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::KDE_SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test single-tree implementation results against brute force results using
 * a cover-tree and Gaussian kernel.
 */
TEST_CASE("GaussianCoverSingleKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::KDE_SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test single-tree implementation results against brute force results using
 * an octree and Epanechnikov kernel.
 */
TEST_CASE("EpanechnikovOctreeSingleKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<EpanechnikovKernel, EuclideanDistance, arma::mat, Octree> kde(
      relError, 0.0, kernel, KDEMode::KDE_SINGLE_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test BallTree dual-tree implementation results against brute force results.
 */
TEST_CASE("BallTreeGaussianKDETest", "[KDETest]")
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
  typedef BallTree<EuclideanDistance, KDEStat, arma::mat> Tree;
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
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));

  delete queryTree;
  delete referenceTree;
}

/**
 * Test Octree dual-tree implementation results against brute force results.
 */
TEST_CASE("OctreeGaussianKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, Octree> kde(
      relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test RTree dual-tree implementation results against brute force results.
 */
TEST_CASE("RTreeGaussianKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, RTree> kde(
      relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test Standard Cover Tree dual-tree implementation results against brute
 * force results using Gaussian kernel.
 */
TEST_CASE("StandardCoverTreeGaussianKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test Standard Cover Tree dual-tree implementation results against brute
 * force results using Epanechnikov kernel.
 */
TEST_CASE("StandardCoverTreeEpanechnikovKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<EpanechnikovKernel,
      EuclideanDistance,
      arma::mat,
      StandardCoverTree>
      kde(relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test duplicated value in reference matrix.
 */
TEST_CASE("DuplicatedReferenceSampleKDETest", "[KDETest]")
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
  typedef KDTree<EuclideanDistance, KDEStat, arma::mat> Tree;
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
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));

  delete queryTree;
  delete referenceTree;
}

/**
 * Test duplicated value in query matrix.
 */
TEST_CASE("DuplicatedQuerySampleKDETest", "[KDETest]")
{
  arma::mat reference = arma::randu(2, 30);
  arma::mat query = arma::randu(2, 10);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.4;
  const double relError = 0.05;

  // Duplicate value.
  query.col(2) = query.col(3);

  // Dual-tree KDE.
  typedef KDTree<EuclideanDistance, KDEStat, arma::mat> Tree;
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
  REQUIRE(estimations[2] == Approx(estimations[3]).epsilon(relError));

  delete queryTree;
  delete referenceTree;
}

/**
 * Test dual-tree breadth-first implementation results against brute force
 * results.
 */
TEST_CASE("BreadthFirstKDETest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree,
      KDTree<EuclideanDistance, KDEStat, arma::mat>::template
          BreadthFirstDualTreeTraverser>
      kde(relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test 1-dimensional implementation results against brute force results.
 */
TEST_CASE("OneDimensionalTest", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kde(
      relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);
  kde.Evaluate(query, treeEstimations);

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(bfEstimations[i] == Approx(treeEstimations[i]).epsilon(relError));
}

/**
 * Test a case where an empty reference set is given to train the model.
 */
TEST_CASE("EmptyReferenceTest", "[KDETest]")
{
  arma::mat reference;
  arma::mat query = arma::randu(1, 10);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // KDE.
  EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kde(
      relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);

  // When training using the dataset matrix.
  REQUIRE_THROWS_AS(kde.Train(reference), std::invalid_argument);

  // When training using a tree.
  std::vector<size_t> oldFromNewReferences;
  typedef KDTree<EuclideanDistance, KDEStat, arma::mat> Tree;
  Tree* referenceTree = new Tree(reference, oldFromNewReferences, 2);
  REQUIRE_THROWS_AS(
      kde.Train(referenceTree, &oldFromNewReferences), std::invalid_argument);

  delete referenceTree;
}

/**
 * Tests when reference set values and query set values dimensions don't match.
 */
TEST_CASE("EvaluationMatchDimensionsTest", "[KDETest]")
{
  arma::mat reference = arma::randu(3, 10);
  arma::mat query = arma::randu(1, 10);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // KDE.
  EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree> kde(relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);

  // When evaluating using the query dataset matrix.
  REQUIRE_THROWS_AS(kde.Evaluate(query, estimations),
                    std::invalid_argument);

  // When evaluating using a query tree.
  typedef KDTree<EuclideanDistance, KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 3);
  REQUIRE_THROWS_AS(kde.Evaluate(queryTree, oldFromNewQueries, estimations),
                    std::invalid_argument);
  delete queryTree;
}

/**
 * Tests when an empty query set is given to be evaluated.
 */
TEST_CASE("EmptyQuerySetTest", "[KDETest]")
{
  arma::mat reference = arma::randu(1, 10);
  arma::mat query;
  // Set estimations to the wrong size.
  arma::vec estimations(33, arma::fill::zeros);
  const double kernelBandwidth = 0.7;
  const double relError = 0.01;

  // KDE.
  EuclideanDistance metric;
  GaussianKernel kernel(kernelBandwidth);
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree> kde(relError, 0.0, kernel, KDEMode::KDE_DUAL_TREE_MODE, metric);
  kde.Train(reference);

  // The query set must be empty.
  REQUIRE(query.n_cols == 0);
  // When evaluating using the query dataset matrix.
  REQUIRE_NOTHROW(kde.Evaluate(query, estimations));

  // When evaluating using a query tree.
  typedef KDTree<EuclideanDistance, KDEStat, arma::mat> Tree;
  std::vector<size_t> oldFromNewQueries;
  Tree* queryTree = new Tree(query, oldFromNewQueries, 3);
  REQUIRE_NOTHROW(
      kde.Evaluate(queryTree, oldFromNewQueries, estimations));
  delete queryTree;

  // Estimations must be empty.
  REQUIRE(estimations.size() == 0);
}

/**
 * Tests serialiation of KDE models.
 */
TEST_CASE("KDESerializationTest", "[KDETest]")
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
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kde(
      relError,
      absError,
      GaussianKernel(0.25),
      KDEMode::KDE_DUAL_TREE_MODE,
      EuclideanDistance(),
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
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kdeXml, kdeText,
      kdeBinary;
  SerializeObjectAll(kde, kdeXml, kdeText, kdeBinary);

  // Check everything is correct.
  REQUIRE(kde.RelativeError() == Approx(relError).epsilon(1e-10));
  REQUIRE(kdeXml.RelativeError() == Approx(relError).epsilon(1e-10));
  REQUIRE(kdeText.RelativeError() == Approx(relError).epsilon(1e-10));
  REQUIRE(kdeBinary.RelativeError() == Approx(relError).epsilon(1e-10));

  REQUIRE(kde.AbsoluteError() == Approx(absError).epsilon(1e-10));
  REQUIRE(kdeXml.AbsoluteError() == Approx(absError).epsilon(1e-10));
  REQUIRE(kdeText.AbsoluteError() == Approx(absError).epsilon(1e-10));
  REQUIRE(kdeBinary.AbsoluteError() == Approx(absError).epsilon(1e-10));

  REQUIRE(kde.IsTrained() == true);
  REQUIRE(kdeXml.IsTrained() == true);
  REQUIRE(kdeText.IsTrained() == true);
  REQUIRE(kdeBinary.IsTrained() == true);

  const KDEMode mode = KDEMode::KDE_DUAL_TREE_MODE;
  REQUIRE(kde.Mode() == mode);
  REQUIRE(kdeXml.Mode() == mode);
  REQUIRE(kdeText.Mode() == mode);
  REQUIRE(kdeBinary.Mode() == mode);

  REQUIRE(kde.MonteCarlo() == monteCarlo);
  REQUIRE(kdeXml.MonteCarlo() == monteCarlo);
  REQUIRE(kdeText.MonteCarlo() == monteCarlo);
  REQUIRE(kdeBinary.MonteCarlo() == monteCarlo);

  REQUIRE(kde.MCProb() == Approx(MCProb).epsilon(1e-10));
  REQUIRE(kdeXml.MCProb() == Approx(MCProb).epsilon(1e-10));
  REQUIRE(kdeText.MCProb() == Approx(MCProb).epsilon(1e-10));
  REQUIRE(kdeBinary.MCProb() == Approx(MCProb).epsilon(1e-10));

  REQUIRE(kde.MCInitialSampleSize() == initialSampleSize);
  REQUIRE(kdeXml.MCInitialSampleSize() == initialSampleSize);
  REQUIRE(kdeText.MCInitialSampleSize() == initialSampleSize);
  REQUIRE(kdeBinary.MCInitialSampleSize() == initialSampleSize);

  REQUIRE(kde.MCEntryCoef() == Approx(entryCoef).epsilon(1e-10));
  REQUIRE(kdeXml.MCEntryCoef() == Approx(entryCoef).epsilon(1e-10));
  REQUIRE(kdeText.MCEntryCoef() == Approx(entryCoef).epsilon(1e-10));
  REQUIRE(kdeBinary.MCEntryCoef() == Approx(entryCoef).epsilon(1e-10));

  REQUIRE(kde.MCBreakCoef() == Approx(breakCoef).epsilon(1e-10));
  REQUIRE(kdeXml.MCBreakCoef() == Approx(breakCoef).epsilon(1e-10));
  REQUIRE(kdeText.MCBreakCoef() == Approx(breakCoef).epsilon(1e-10));
  REQUIRE(kdeBinary.MCBreakCoef() == Approx(breakCoef).epsilon(1e-10));

  // Test if execution gives the same result.
  arma::vec xmlEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec textEstimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec binEstimations = arma::vec(query.n_cols, arma::fill::zeros);

  kdeXml.Evaluate(query, xmlEstimations);
  kdeText.Evaluate(query, textEstimations);
  kdeBinary.Evaluate(query, binEstimations);

  for (size_t i = 0; i < query.n_cols; ++i)
  {
    REQUIRE(estimations[i] == Approx(xmlEstimations[i]).epsilon(relError));
    REQUIRE(estimations[i] == Approx(textEstimations[i]).epsilon(relError));
    REQUIRE(estimations[i] == Approx(binEstimations[i]).epsilon(relError));
  }
}

/**
 * Test if the copy constructor and copy operator works properly.
 */
TEST_CASE("CopyConstructor", "[KDETest]")
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec estimations1, estimations2, estimations3;
  const double kernelBandwidth = 1.5;
  const double relError = 0.05;

  typedef KDE<GaussianKernel, EuclideanDistance, arma::mat> KDEType;

  // KDE.
  KDEType kde(relError, 0, GaussianKernel(kernelBandwidth));
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
    REQUIRE(estimations1[i] == Approx(estimations2[i]).epsilon(1e-12));
    REQUIRE(estimations2[i] == Approx(estimations3[i]).epsilon(1e-12));
  }
}

/**
 * Test if the move constructor works properly.
 */
TEST_CASE("MoveConstructor", "[KDETest]")
{
  arma::mat reference = arma::randu(2, 300);
  arma::mat query = arma::randu(2, 100);
  arma::vec estimations1, estimations2, estimations3;
  const double kernelBandwidth = 1.2;
  const double relError = 0.05;

  typedef KDE<EpanechnikovKernel, EuclideanDistance, arma::mat>
      KDEType;

  // KDE.
  KDEType kde(relError, 0, EpanechnikovKernel(kernelBandwidth));
  kde.Train(std::move(reference));
  kde.Evaluate(query, estimations1);

  // Move constructor KDE.
  KDEType constructor(std::move(kde));
  constructor.Evaluate(query, estimations2);

  // Check results.
  REQUIRE_THROWS_AS(kde.Evaluate(query, estimations3), std::runtime_error);
  for (size_t i = 0; i < query.n_cols; ++i)
    REQUIRE(estimations1[i] == Approx(estimations2[i]).epsilon(1e-12));
}

/**
 * Test if an untrained KDE works properly.
 */
TEST_CASE("NotTrained", "[KDETest]")
{
  arma::mat query = arma::randu(1, 10);
  std::vector<size_t> oldFromNew;
  arma::vec estimations;

  KDE<> kde;
  KDE<>::Tree queryTree(query, oldFromNew);

  // Check results.
  REQUIRE_THROWS_AS(kde.Evaluate(query, estimations), std::runtime_error);
  REQUIRE_THROWS_AS(kde.Evaluate(&queryTree, oldFromNew, estimations),
      std::runtime_error);
  REQUIRE_THROWS_AS(kde.Evaluate(estimations), std::runtime_error);
}

/**
 * Test single KD-tree implementation results against brute force results using
 * Monte Carlo estimations when possible.
 */
TEST_CASE("GaussianSingleKDTreeMonteCarloKDE", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kde(
      relError,
      0.0,
      kernel,
      KDEMode::KDE_SINGLE_TREE_MODE,
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

  REQUIRE(correctResults > 70);
}

/**
 * Test single cover-tree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
TEST_CASE("GaussianSingleCoverTreeMonteCarloKDE", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, StandardCoverTree>
      kde(relError,
          0.0,
          kernel,
          KDEMode::KDE_SINGLE_TREE_MODE,
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

  REQUIRE(correctResults > 70);
}

/**
 * Test single octree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
TEST_CASE("GaussianSingleOctreeMonteCarloKDE", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, Octree> kde(
      relError,
      0.0,
      kernel,
      KDEMode::KDE_SINGLE_TREE_MODE,
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

  REQUIRE(correctResults > 70);
}

/**
 * Test dual kd-tree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
TEST_CASE("GaussianDualKDTreeMonteCarloKDE", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, KDTree> kde(
      relError,
      0.0,
      kernel,
      KDEMode::KDE_DUAL_TREE_MODE,
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

  REQUIRE(correctResults > 70);
}

/**
 * Test dual Cover-tree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
TEST_CASE("GaussianDualCoverTreeMonteCarloKDE", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, StandardCoverTree>
      kde(relError,
          0.0,
          kernel,
          KDEMode::KDE_DUAL_TREE_MODE,
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

  REQUIRE(correctResults > 70);
}

/**
 * Test dual octree implementation results against brute force results
 * using Monte Carlo estimations when possible.
 */
TEST_CASE("GaussianDualOctreeMonteCarloKDE", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel, EuclideanDistance, arma::mat, Octree> kde(
      relError,
      0.0,
      kernel,
      KDEMode::KDE_DUAL_TREE_MODE,
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

  REQUIRE(correctResults > 70);
}

/**
 * Test dual kd-tree breadth first traversal implementation results against
 * brute force results using Monte Carlo estimations when possible.
 */
TEST_CASE("GaussianBreadthDualKDTreeMonteCarloKDE", "[KDETest]")
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
  EuclideanDistance metric;
  KDE<GaussianKernel,
      EuclideanDistance,
      arma::mat,
      KDTree,
      KDTree<EuclideanDistance, KDEStat, arma::mat>::template
          BreadthFirstDualTreeTraverser>
    kde(relError,
        0.0,
        kernel,
        KDEMode::KDE_DUAL_TREE_MODE,
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

  REQUIRE(correctResults > 70);
}
