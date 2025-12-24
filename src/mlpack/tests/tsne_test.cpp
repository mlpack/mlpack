/**
 * @file tests/tsne_test.cpp
 * @author Ranjodh Singh
 *
 * t-SNE Tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

#include "../methods/tsne/tsne.hpp"
#include "../methods/tsne/tsne_utils.hpp"

using namespace mlpack;

/**
 * Test the functions that can modify and access the parameters of
 * the t-SNE Class.
 */
TEST_CASE("TSNEParameterTest", "[TSNETest]")
{
  TSNE tsne(2, 30.0, 12.0, 200.0, 1000, 1e-12, "pca", 0.5);

  // Make sure we can get the parameters successfully.
  REQUIRE(tsne.OutputDimensions() == 2);
  REQUIRE(tsne.Perplexity() == 30.0);
  REQUIRE(tsne.Exaggeration() == 12.0);
  REQUIRE(tsne.StepSize() == 200.0);
  REQUIRE(tsne.Tolerance() == 1e-12);
  REQUIRE(tsne.MaximumIterations() == 1000);
  REQUIRE(tsne.Initialization() == "pca");
  REQUIRE(tsne.Theta() == 0.5);

  // Now modify the parameters to match the second layer.
  tsne.OutputDimensions() = 3;
  tsne.Perplexity() = 50.0;
  tsne.Exaggeration() = 4.0;
  tsne.StepSize() = 50.0;
  tsne.Tolerance() = 1e-9;
  tsne.MaximumIterations() = 400;
  tsne.Initialization() = "random";
  tsne.Theta() = 0.3;

  // Make sure we can get the parameters successfully.
  REQUIRE(tsne.OutputDimensions() == 3);
  REQUIRE(tsne.Perplexity() == 50.0);
  REQUIRE(tsne.Exaggeration() == 4.0);
  REQUIRE(tsne.StepSize() == 50.0);
  REQUIRE(tsne.Tolerance() == 1e-9);
  REQUIRE(tsne.MaximumIterations() == 400);
  REQUIRE(tsne.Initialization() == "random");
  REQUIRE(tsne.Theta() == 0.3);
}

/**
 * Test t-SNE PCA Initialization
 */
TEST_CASE("TSNEInitializationTest", "[TSNETest]")
{
  using MatType = arma::mat;
  using VecType = GetColType<MatType>::type;

  const size_t nDims = 2;
  const size_t nPoints = 10;
  TSNE tsne(nDims, 30.0, 12.0, 200.0, 1000, 1e-12, "pca", 0.5);

  MatType X(nDims, nPoints, arma::fill::randn), Y;
  tsne.InitializeEmbedding(X, Y);

  VecType stdDev = stddev(Y, 0, 1);
  for (size_t i = 0; i < nDims; i++)
    REQUIRE(stdDev[i] == Approx(1e-4));

  tsne.Initialization() = "random";
  tsne.InitializeEmbedding(X, Y);

  stdDev = stddev(Y, 0, 1);
  for (size_t i = 0; i < nDims; i++)
    REQUIRE(stdDev[i] == Approx(1e-4));
}

/**
 * Test t-SNE Exact Input Joint Probabilities Computation
 */
TEST_CASE("TSNEExactPCalcTest", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  const double desiredPerplexity = 30.0;
  MatType D = PairwiseDistances(X, SquaredEuclideanDistance());
  MatType P = computeInputProbabilities(30.0, D, false);
  const double meanPerplexity = mean(exp(-sum(
      P % log(clamp(P, arma::datum::eps, arma::datum::inf)), 1)));

  REQUIRE(meanPerplexity == Approx(desiredPerplexity).margin(1e-6));
}

/**
 * Test t-SNE Approximate Input Joint Probabilities Computation
 */
TEST_CASE("TSNEApproxPCalcTest", "[TSNETest]")
{
  using MatType = arma::mat;
  using UMatType = arma::Mat<size_t>;
  using VecType = GetColType<MatType>::type;
  using SpMatType = GetSparseMatType<MatType>::type;

  MatType X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  const double desiredPerplexity = 3.0;

  const size_t n = X.n_cols;
  const size_t k = std::min<size_t>(
      (size_t)(n - 1), (size_t)(3 * desiredPerplexity));

  MatType distances;
  UMatType neighbors;
  NeighborSearch<
      NearestNeighborSort,
      SquaredEuclideanDistance,
      MatType
  > knn(X);
  knn.Search(k, neighbors, distances);

  SpMatType P = computeInputProbabilities(
      desiredPerplexity, neighbors, distances, false);

  VecType perplexities(n);
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      const double p = (double) P(i, neighbors(j, i));
      if (p)
        perplexities(i) += p * std::log(p);
    }
    perplexities(i) = std::exp(-perplexities(i));
  }
  const double meanPerplexity = mean(perplexities);

  REQUIRE(meanPerplexity == Approx(desiredPerplexity).margin(1e-6));
}

/**
 * Test t-SNE Exact Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEExactIris", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<ExactTSNE, MatType> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEBarnesHutIris", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<BarnesHutTSNE, MatType> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEDualTreeIris", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<DualTreeTSNE, MatType> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Exact Method Final Error on Iris dataset with fmat.
 */
TEST_CASE("TSNEExactIrisFmat", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<ExactTSNE, MatType> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset with fmat.
 */
TEST_CASE("TSNEBarnesHutIrisFmat", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<BarnesHutTSNE, MatType> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset with fmat.
 */
TEST_CASE("TSNEDualTreeIrisFmat", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<DualTreeTSNE, MatType> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Exact Method Final Error on Iris dataset with fmat
 * and ManhattanDistance.
 */
TEST_CASE("TSNEExactIrisFmatManhattan", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<ExactTSNE, MatType, ManhattanDistance> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset with fmat
 * and ManhattanDistance.
 */
TEST_CASE("TSNEBarnesHutIrisFmatManhattan", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<BarnesHutTSNE, MatType, ManhattanDistance> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset with fmat
 * and ManhattanDistance.
 */
TEST_CASE("TSNEDualTreeIrisFmatManhattan", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<DualTreeTSNE, MatType, ManhattanDistance> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Exact Method Final Error on Iris dataset with fmat and
 * EuclideanDistance.
 */
TEST_CASE("TSNEExactIrisFmatEuclidean", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<ExactTSNE, MatType, EuclideanDistance> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset with fmat and
 * EuclideanDistance.
 */
TEST_CASE("TSNEBarnesHutIrisFmatEuclidean", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<BarnesHutTSNE, MatType, EuclideanDistance> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset with fmat and
 * EuclideanDistance.
 */
TEST_CASE("TSNEDualTreeIrisFmatEuclidean", "[TSNETest]")
{
  using MatType = arma::fmat;

  MatType X, Y;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<DualTreeTSNE, MatType, EuclideanDistance> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  const double finalObjective = tsne.Embed(X, Y);
  std::cout << finalObjective << std::endl;
}

/**
 * Test whether BarnesHutTSNE with theta == 0 matches ExactTSNE.
 */
TEST_CASE("TSNEBarnesHutMatchExact", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X, YExact, YBarnesHut;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<ExactTSNE, MatType> tsneExact(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);
  TSNE<BarnesHutTSNE, MatType> tsneBarnesHut(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);

  const double finalObjectiveExact = tsneExact.Embed(X, YExact);
  const double finalObjectiveBarnesHut = tsneBarnesHut.Embed(X, YBarnesHut);

  std::cout << finalObjectiveExact << ' '
            << finalObjectiveBarnesHut << std::endl;

  REQUIRE(finalObjectiveExact ==
          Approx(finalObjectiveBarnesHut).margin(1e-3));
  REQUIRE(arma::approx_equal(YExact, YBarnesHut, "absdiff", 1e-3));
}

/**
 * Test whether DualTreeTSNE with theta == 0 matches BarnesHutTSNE
 * with theta == 0.
 */
TEST_CASE("TSNEDualTreeMatchBarnesHut", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X, YDualTree, YBarnesHut;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<DualTreeTSNE, MatType> tsneDualTree(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);
  TSNE<BarnesHutTSNE, MatType> tsneBarnesHut(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);

  const double finalObjectiveDualTree = tsneDualTree.Embed(X, YDualTree);
  const double finalObjectiveBarnesHut = tsneBarnesHut.Embed(X, YBarnesHut);

  std::cout << finalObjectiveDualTree << ' '
            << finalObjectiveBarnesHut << std::endl;

  REQUIRE(finalObjectiveDualTree ==
          Approx(finalObjectiveBarnesHut).margin(1e-3));
  REQUIRE(arma::approx_equal(YDualTree, YBarnesHut, "absdiff", 1e-3));
}

#ifdef MLPACK_USE_OPENMP

/**
 * Verify that BarnesHutTSNE produces approximately equal results
 * when executed in parallel and in sequential modes.
 */
TEST_CASE("TSNEBHMultithreadDeterminismTest", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X, YSeq, YParallel;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  const size_t maxThreads = omp_get_max_threads();
  omp_set_num_threads(1);

  TSNE<BarnesHutTSNE, MatType> tsneSeq(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);
  const double finalObjectiveSeq = tsneSeq.Embed(X, YSeq);

  omp_set_num_threads(maxThreads);

  TSNE<BarnesHutTSNE, MatType> tsneParallel(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);
  const double finalObjectiveParallel = tsneParallel.Embed(X, YParallel);

  std::cout << finalObjectiveSeq << ' '
            << finalObjectiveParallel << std::endl;

  REQUIRE(finalObjectiveSeq ==
          Approx(finalObjectiveParallel).margin(1e-3));
  REQUIRE(arma::approx_equal(YParallel, YSeq, "absdiff", 1e-3));
}

/**
 * Verify that DualTreeTSNE produces approximately equal results
 * when executed in parallel and in sequential modes.
 */
TEST_CASE("TSNEDualTreeMultithreadDeterminismTest", "[TSNETest]")
{
  using MatType = arma::mat;

  MatType X, YSeq, YParallel;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  const size_t maxThreads = omp_get_max_threads();
  omp_set_num_threads(1);

  TSNE<DualTreeTSNE, MatType> tsneSeq(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);
  const double finalObjectiveSeq = tsneSeq.Embed(X, YSeq);

  omp_set_num_threads(maxThreads);

  TSNE<DualTreeTSNE, MatType> tsneParallel(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);
  const double finalObjectiveParallel = tsneParallel.Embed(X, YParallel);

  std::cout << finalObjectiveSeq << ' '
            << finalObjectiveParallel << std::endl;

  REQUIRE(finalObjectiveSeq ==
          Approx(finalObjectiveParallel).margin(1e-3));
  REQUIRE(arma::approx_equal(YParallel, YSeq, "absdiff", 1e-3));
}

#endif

// /* Uniform grid recovery */
// TEST_CASE("TSNEUniformGridRecoveryTest", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Use a 2D uniform grid embedded in higher-D (X_2d_grid equivalent).
//   // - Run TSNE (several seeds) and check nearest-neighbor spacing: min
//   // distance > 0.1,
//   //   smallest/mean and largest/mean ratios in acceptable bounds.
//   // - If first run fails, rerun using previous embedding as init and
//   re-check. SUCCEED();
// }

// /* Trustworthiness tests */
// TEST_CASE("TSNETrustworthinessTest", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Test trustworthiness for:
//   //   * Affine transform -> trustworthiness == 1.0
//   //   * Random permutation -> trustworthiness < 0.6
//   //   * Small controlled permutation -> exact numeric trustworthiness
//   // - Also implement n_neighbors validation throwing when invalid.
//   SUCCEED();
// }

// /* Preserve trustworthiness approximately (exact/barnes_hut + init
// random/pca)
//  */
// TEST_CASE("TSNEPreserveTrustworthinessApproximatelyTest", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - For both methods ('exact', 'barnes_hut') and both inits ('random',
//   // 'pca'):
//   //   Run TSNE on random X and assert trustworthiness > 0.85 for
//   //   n_neighbors=1.
//   SUCCEED();
// }

// /** Gradient descent stopping conditions (mapping
// test_gradient_descent_stops)
//  */
// TEST_CASE("TSNEGradientDescentStopsTest", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Port the ObjectiveSmallGradient and flat_function behaviours:
//   //   * Check stopping on min_grad_norm triggers "gradient norm" message
//   (if
//   //   logging exists).
//   //   * Check n_iter_without_progress behaviour triggers appropriate stop
//   and
//   //   message.
//   //   * Check max_iter stops at expected iteration count.
//   // - If your _gradient_descent writes to stdout, capture and assert
//   messages. SUCCEED();
// }

// /* n_iter_without_progress and min_grad_norm behaviour tests */
// TEST_CASE("TSNENIterWithoutProgressAndMinGradNormTests", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Test edge cases for n_iter_without_progress negative handling and
//   // min_grad_norm.
//   // - Assert the verbose output includes "did not make any progress"
//   message
//   // when appropriate.
//   SUCCEED();
// }

// /* Precomputed distances usage and validations */
// TEST_CASE("TSNEPrecomputedDistancesValidationTest", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Validate errors are raised for:
//   //   * non-square distance matrices,
//   //   * non-positive distances,
//   //   * sparse precomputed distances with 'exact' method.
//   SUCCEED();
// }
