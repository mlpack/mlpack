/**
 * @file tests/regularized_svd_test.cpp
 * @author Ranjodh Singh
 *
 * t-SNE Tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/distances/mahalanobis_distance.hpp>
#include <omp.h>
#include <limits>
#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "./catch.hpp"
#include "./test_catch_tools.hpp"

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
TEST_CASE("TSNEPCAInitTest", "[TSNETest]")
{
  const size_t nDims = 2;
  const size_t nPoints = 10;
  TSNE tsne(nDims, 30.0, 12.0, 200.0, 1000, 1e-12, "pca", 0.5);

  arma::mat X(nDims, nPoints, arma::fill::randn), Y;
  tsne.InitializeEmbedding(X, Y);

  arma::vec stdDev = stddev(Y, 0, 1);
  for (size_t i = 0; i < nDims; i++)
    REQUIRE(stdDev[i] == Approx(1e-4));
}

/**
 * Test t-SNE Random Initialization
 */
TEST_CASE("TSNERandomInitTest", "[TSNETest]")
{
  const size_t nDims = 2;
  const size_t nPoints = 10;
  TSNE tsne(nDims, 30.0, 12.0, 200.0, 1000, 1e-12, "random", 0.5);

  arma::mat X(nDims, nPoints, arma::fill::randn), Y;
  tsne.InitializeEmbedding(X, Y);

  arma::vec stdDev = stddev(Y, 0, 1);
  for (size_t i = 0; i < nDims; i++)
    REQUIRE(stdDev[i] == Approx(1e-4).margin(1e-6));
}

/**
 * Test t-SNE Exact Input Joint Probabilities Computation
 */
TEST_CASE("TSNEExactPCalcTest", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  const double desiredPerplexity = 30.0;
  arma::mat D = PairwiseDistances(X, SquaredEuclideanDistance());
  arma::mat P = computeInputProbabilities(30.0, D, false);
  const double meanPerplexity = mean(exp(-sum(
      P % log(clamp(P, arma::datum::eps, arma::datum::inf)), 1)));

  REQUIRE(meanPerplexity == Approx(desiredPerplexity).margin(1e-6));
}

/**
 * Test t-SNE Approximate Input Joint Probabilities Computation
 */
TEST_CASE("TSNEApproxPCalcTest", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  const double desiredPerplexity = 3.0;

  const size_t n = X.n_cols;
  const size_t k = std::min<size_t>(
      (size_t)(n - 1), (size_t)(3 * desiredPerplexity));

  arma::mat distances;
  arma::Mat<size_t> neighbors;
  NeighborSearch<
      NearestNeighborSort,
      SquaredEuclideanDistance,
      arma::mat
  > knn(X);
  knn.Search(k, neighbors, distances);

  arma::sp_mat P = computeInputProbabilities(
      desiredPerplexity, neighbors, distances, false);

  arma::vec perplexities(n);
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
  // To Do: Verify KL at various points.
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::mat, SquaredEuclideanDistance, ExactTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::mat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEBarnesHutIris", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::mat, SquaredEuclideanDistance, BarnesHutTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::mat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEDualTreeIris", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::mat, SquaredEuclideanDistance, DualTreeTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::mat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/* Barnes-Hut should match Exact under specific conditions. */
TEST_CASE("TSNEBarnesHutMatchExact", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::mat, SquaredEuclideanDistance, ExactTSNE> tsneExact(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);

  arma::mat YExact;
  const double finalObjectiveExact = tsneExact.Embed(X, YExact);

  TSNE<arma::mat, SquaredEuclideanDistance, BarnesHutTSNE> tsneBarnesHut(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);

  arma::mat YBarnesHut;
  const double finalObjectiveBarnesHut = tsneBarnesHut.Embed(X, YBarnesHut);

  std::cout << finalObjectiveBarnesHut << ' '
            << finalObjectiveExact << std::endl;

  // REQUIRE(finalObjectiveBarnesHut ==
  //         Approx(finalObjectiveExact).margin(1e-3));
  // REQUIRE(arma::approx_equal(YExact, YBarnesHut, "absdiff", 1e-3));
}

/* Dual-tree angle == 0 should match Barnes-Hut */
TEST_CASE("TSNEDualTreeAngleZeroMatchBarnesHut", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::mat, SquaredEuclideanDistance, DualTreeTSNE> tsneDualTree(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);

  arma::mat YDualTree;
  const double finalObjectiveDualTree = tsneDualTree.Embed(X, YDualTree);

  TSNE<arma::mat, SquaredEuclideanDistance, BarnesHutTSNE> tsneBarnesHut(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.0);

  arma::mat YBarnesHut;
  const double finalObjectiveBarnesHut = tsneBarnesHut.Embed(X, YBarnesHut);

  std::cout << finalObjectiveBarnesHut << ' '
            << finalObjectiveDualTree << std::endl;

  // REQUIRE(finalObjectiveDualTree ==
  //         Approx(finalObjectiveBarnesHut).margin(1e-3));
  // REQUIRE(arma::approx_equal(YDualTree, YBarnesHut, "absdiff", 1e-3));
}

/* Barnes-Hut gradient multi-threading determinism */
TEST_CASE("TSNEGradientBHMultithreadDeterminismTest", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  // Initial max threads
  const size_t maxThreads = omp_get_max_threads();

  // Sequential Mode
  omp_set_num_threads(1);

  TSNE<arma::mat, SquaredEuclideanDistance, BarnesHutTSNE> tsneSeq(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::mat YSeq;
  const double finalObjectiveSeq = tsneSeq.Embed(X, YSeq);

  // Parallel Mode
  omp_set_num_threads(maxThreads);

  TSNE<arma::mat, SquaredEuclideanDistance, BarnesHutTSNE> tsneParallel(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::mat YParallel;
  const double finalObjectiveParallel = tsneParallel.Embed(X, YParallel);

  std::cout << finalObjectiveSeq << ' ' << finalObjectiveParallel << std::endl;

  // REQUIRE(finalObjectiveParallel ==
  //         Approx(finalObjectiveSeq).margin(1e-3));
  // REQUIRE(arma::approx_equal(YParallel, YSeq, "absdiff", 1e-3));
}

/* Dual-Tree gradient multi-threading determinism */
TEST_CASE("TSNEGradientDualTreeMultithreadDeterminismTest", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  // Initial max threads
  const size_t maxThreads = omp_get_max_threads();

  // Sequential Mode
  omp_set_num_threads(1);

  TSNE<arma::mat, SquaredEuclideanDistance, DualTreeTSNE> tsneSeq(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::mat YSeq;
  const double finalObjectiveSeq = tsneSeq.Embed(X, YSeq);

  // Parallel Mode
  omp_set_num_threads(maxThreads);

  TSNE<arma::mat, SquaredEuclideanDistance, DualTreeTSNE> tsneParallel(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::mat YParallel;
  const double finalObjectiveParallel = tsneParallel.Embed(X, YParallel);

  std::cout << finalObjectiveSeq << ' ' << finalObjectiveParallel << std::endl;

  // REQUIRE(finalObjectiveParallel ==
  //         Approx(finalObjectiveSeq).margin(1e-3));
  // REQUIRE(arma::approx_equal(YParallel, YSeq, "absdiff", 1e-3));
}

/**
 * Test t-SNE Exact Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEExactIrisFloat", "[TSNETest]")
{
  // To Do: Verify KL at various points.
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, ExactTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEBarnesHutIrisFloat", "[TSNETest]")
{
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, BarnesHutTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEDualTreeIrisFloat", "[TSNETest]")
{
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, DualTreeTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Exact Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEExactIrisFloatManhattan", "[TSNETest]")
{
  // To Do: Verify KL at various points.
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, EuclideanDistance, ExactTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEBarnesHutIrisFloatEuclidean", "[TSNETest]")
{
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, ManhattanDistance, BarnesHutTSNE> tsne(
      2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}


/**
 * Test t-SNE Exact Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEExactIrisFloat1D", "[TSNETest]")
{
  // To Do: Verify KL at various points.
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, ExactTSNE> tsne(
      1, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEBarnesHutIrisFloat1D", "[TSNETest]")
{
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, BarnesHutTSNE> tsne(
      1, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEDualTreeIrisFloat1D", "[TSNETest]")
{
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, DualTreeTSNE> tsne(
      1, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Exact Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEExactIrisFloat3D", "[TSNETest]")
{
  // To Do: Verify KL at various points.
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, ExactTSNE> tsne(
      3, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEBarnesHutIrisFloat3D", "[TSNETest]")
{
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, BarnesHutTSNE> tsne(
      3, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEDualTreeIrisFloat3D", "[TSNETest]")
{
  arma::fmat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<arma::fmat, SquaredEuclideanDistance, DualTreeTSNE> tsne(
      3, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

  arma::fmat Y;
  const double finalObjective = tsne.Embed(X, Y);

  std::cout << finalObjective << std::endl;
}

// /**
//  * Test t-SNE Dual-Tree Method Final Error on Iris dataset.
//  */
// TEST_CASE("TSNEDualTreeIrisFloatChebyshev", "[TSNETest]")
// {
//   arma::fmat X;
//   if (!data::Load("iris.csv", X))
//     FAIL("Cannot load test dataset iris.csv!");

//   TSNE<arma::fmat, ChebyshevDistance, DualTreeTSNE> tsne(
//       2, 30.0, 12.0, 0.0, 500, 1e-12, "pca", 0.5);

//   arma::fmat Y;
//   const double finalObjective = tsne.Embed(X, Y);

//   std::cout << finalObjective << std::endl;
// }

// /* Uniform grid recovery */
// TEST_CASE("TSNEUniformGridRecoveryTest", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Use a 2D uniform grid embedded in higher-D (X_2d_grid equivalent).
//   // - Run TSNE (several seeds) and check nearest-neighbor spacing: min
//   // distance > 0.1,
//   //   smallest/mean and largest/mean ratios in acceptable bounds.
//   // - If first run fails, rerun using previous embedding as init and re-check.
//   SUCCEED();
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

// /* Preserve trustworthiness approximately (exact/barnes_hut + init random/pca)
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

// /** Gradient descent stopping conditions (mapping test_gradient_descent_stops)
//  */
// TEST_CASE("TSNEGradientDescentStopsTest", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Port the ObjectiveSmallGradient and flat_function behaviours:
//   //   * Check stopping on min_grad_norm triggers "gradient norm" message (if
//   //   logging exists).
//   //   * Check n_iter_without_progress behaviour triggers appropriate stop and
//   //   message.
//   //   * Check max_iter stops at expected iteration count.
//   // - If your _gradient_descent writes to stdout, capture and assert messages.
//   SUCCEED();
// }

// /* n_iter_without_progress and min_grad_norm behaviour tests */
// TEST_CASE("TSNENIterWithoutProgressAndMinGradNormTests", "[TSNETest]")
// {
//   // IMPLEMENT:
//   // - Test edge cases for n_iter_without_progress negative handling and
//   // min_grad_norm.
//   // - Assert the verbose output includes "did not make any progress" message
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