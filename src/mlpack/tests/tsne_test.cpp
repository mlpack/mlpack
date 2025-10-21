/**
 * @file tests/regularized_svd_test.cpp
 * @author Ranjodh Singh
 *
 * Tests for the t-distributed stochastic neighbor embedding implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <omp.h>
#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "../methods/tsne/tsne.hpp"
#include "../methods/tsne/tsne_utils.hpp"

using namespace mlpack;

/**
 * Test functions that can modify and access the parameters of the TSNE Class.
 */
TEST_CASE("TSNEParameterTest", "[TSNETest]")
{
  TSNE tsne(2, 30.0, 12.0, 200.0, 1000, "pca", 0.5);

  // Make sure we can get the parameters successfully.
  REQUIRE(tsne.OutputDimensions() == 2);
  REQUIRE(tsne.Perplexity() == 30.0);
  REQUIRE(tsne.Exaggeration() == 12.0);
  REQUIRE(tsne.StepSize() == 200.0);
  REQUIRE(tsne.MaximumIterations() == 1000);
  REQUIRE(tsne.Initialization() == "pca");
  REQUIRE(tsne.Theta() == 0.5);

  // Now modify the parameters to match the second layer.
  tsne.OutputDimensions() = 3;
  tsne.Perplexity() = 50.0;
  tsne.Exaggeration() = 4.0;
  tsne.StepSize() = 50.0;
  tsne.MaximumIterations() = 400;
  tsne.Initialization() = "random";
  tsne.Theta() = 0.3;

  // Make sure we can get the parameters successfully.
  REQUIRE(tsne.OutputDimensions() == 3);
  REQUIRE(tsne.Perplexity() == 50.0);
  REQUIRE(tsne.Exaggeration() == 4.0);
  REQUIRE(tsne.StepSize() == 50.0);
  REQUIRE(tsne.MaximumIterations() == 400);
  REQUIRE(tsne.Initialization() == "random");
  REQUIRE(tsne.Theta() == 0.3);
}

// /**
//  * Test t-SNE PCA Initialization
//  */
// TEST_CASE("TSNEPCAInitTest", "[TSNETest]") {}

// /**
//  * Test t-SNE Random Initialization
//  */
// TEST_CASE("TSNERandomInitTest", "[TSNETest]") {}

// /**
//  * Test t-SNE Exact Input Joint Probabilities Computation
//  */
// TEST_CASE("TSNEExactPCalcTest", "[TSNETest]")
// {
//   // ToDo: Also include underflow / stability case with small float32 values.

//   arma::mat X;
//   if (!data::Load("iris.csv", X))
//     FAIL("Cannot load test dataset iris.csv!");

//   const double desiredPerplexity = 30.0;
//   arma::mat D = PairwiseDistances(X, SquaredEuclideanDistance());
//   arma::mat P = computeInputJointProbabilities(30.0, D);
//   const double meanPerplexity = arma::mean(arma::exp(
//       -arma::sum(P % arma::log(arma::clamp(P, arma::datum::eps, arma::datum::inf)), 1)));

//   REQUIRE(meanPerplexity == Approx(desiredPerplexity).margin(1e-3));
// }

// /**
//  * Test t-SNE Approximate Input Joint Probabilities Computation
//  */
// TEST_CASE("TSNEApproxPCalcTest", "[TSNETest]")
// {
//   // ToDo: Also include underflow / stability case with small float32 values.

//   arma::mat X;
//   if (!data::Load("iris.csv", X))
//     FAIL("Cannot load test dataset iris.csv!");

//   const double desiredPerplexity = 30.0;
//   const size_t k = std::min<size_t>(X.n_cols - 1, 3 * desiredPerplexity);

//   arma::Mat<size_t> neighbors;
//   arma::Mat<double> distances;
//   NeighborSearch<NearestNeighborSort, SquaredEuclideanDistance> knn(X);
//   knn.Search(k, neighbors, distances);

//   arma::SpMat<double> P = computeInputJointProbabilities(desiredPerplexity,
//                                                          neighbors,
//                                                          distances);

//   arma::vec perplexities(P.n_cols);
//   for (size_t i = 0; i < P.n_cols; i++)
//   {
//     for (size_t j = 0; j < k; j++)
//     {
//       const double p = P(i, neighbors(j, i));
//       perplexities(i) += p * std::log(p);
//     }
//     perplexities(i) = std::exp(-perplexities(i));
//   }
//   const double meanPerplexity = arma::mean(perplexities);

//   REQUIRE(meanPerplexity == Approx(desiredPerplexity).margin(1e-3));
// }

/**
 * Test t-SNE Exact Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEExactIris", "[TSNETest]")
{
  // To Do: Verify KL at various points using callbacks

  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  // 0.126869
  TSNE<ExactTSNE> tsne(2, 30.0, 12.0, 0.0, 500, "pca", 0.5);

  arma::mat Y;
  const double finalObjective = tsne.Embed(X, Y);

  REQUIRE(finalObjective <= Approx(0.126869).margin(1e-5));
}

/**
 * Test t-SNE Barnes-hut Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEBarnesHutIris", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  // 0.118577
  TSNE<BarnesHutTSNE> tsne(2, 30.0, 12.0, 0.0, 500, "pca", 0.5);

  arma::mat Y;
  const double finalObjective = tsne.Embed(X, Y);

  REQUIRE(finalObjective <= Approx(0.118577).margin(1e-1));
}

/**
 * Test t-SNE Dual-Tree Method Final Error on Iris dataset.
 */
TEST_CASE("TSNEDualTreeIris", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<DualTreeTSNE> tsne(2, 30.0, 12.0, 0.0, 500, "pca", 0.5);

  arma::mat Y;
  const double finalObjective = tsne.Embed(X, Y);

  REQUIRE(finalObjective <= Approx(0.118577).margin(1e-1));
}

/* Barnes-Hut should match Exact under specific conditions. */
TEST_CASE("TSNEBarnesHutMatchExact", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  // 0.068447
  TSNE<ExactTSNE> tsneExact(2, 50.0, 12.0, 0.0, 500, "pca", 0.0);
  arma::mat YExact;
  const double finalObjectiveExact = tsneExact.Embed(X, YExact);

  // 0.068561
  TSNE<BarnesHutTSNE> tsneBarnesHut(2, 50.0, 12.0, 0.0, 500, "pca", 0.0);
  arma::mat YBarnesHut;
  const double finalObjectiveBarnesHut = tsneBarnesHut.Embed(X, YBarnesHut);

  REQUIRE(finalObjectiveBarnesHut ==
          Approx(finalObjectiveExact).margin(1e-2));
  // REQUIRE(arma::approx_equal(YExact, YBarnesHut, "absdiff", 1e-2));
}

/* Dual-tree angle == 0 should match Barnes-Hut */
TEST_CASE("TSNEDualTreeAngleZeroMatchBarnesHut", "[TSNETest]")
{
  arma::mat X;
  if (!data::Load("iris.csv", X))
    FAIL("Cannot load test dataset iris.csv!");

  TSNE<DualTreeTSNE> tsneDualTree(2, 30.0, 12.0, 0.0, 500, "pca", 0.0);
  arma::mat YDualTree;
  const double finalObjectiveDualTree = tsneDualTree.Embed(X, YDualTree);

  // 0.128430
  TSNE<BarnesHutTSNE> tsneBarnesHut(2, 30.0, 12.0, 0.0, 500, "pca", 0.0);
  arma::mat YBarnesHut;
  const double finalObjectiveBarnesHut = tsneBarnesHut.Embed(X, YBarnesHut);

  REQUIRE(finalObjectiveDualTree ==
          Approx(finalObjectiveBarnesHut).margin(1e-2));
  // REQUIRE(arma::approx_equal(YDualTree, YBarnesHut, "absdiff", 1e-2));
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

  TSNE<BarnesHutTSNE> tsneSeq(2, 30.0, 12.0, 0.0, 500, "pca", 0.5);
  arma::mat YSeq;
  const double finalObjectiveSeq = tsneSeq.Embed(X, YSeq);

  // Parallel Mode
  omp_set_num_threads(maxThreads);

  TSNE<BarnesHutTSNE> tsneParallel(2, 30.0, 12.0, 0.0, 500, "pca", 0.5);
  arma::mat YParallel;
  const double finalObjectiveParallel = tsneParallel.Embed(X, YParallel);

  REQUIRE(finalObjectiveParallel == Approx(finalObjectiveSeq).margin(1e-3));
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

  TSNE<DualTreeTSNE> tsneSeq(2, 30.0, 12.0, 0.0, 500, "pca", 0.5);
  arma::mat YSeq;
  const double finalObjectiveSeq = tsneSeq.Embed(X, YSeq);

  // Parallel Mode
  omp_set_num_threads(maxThreads);

  TSNE<DualTreeTSNE> tsneParallel(2, 30.0, 12.0, 0.0, 500, "pca", 0.5);
  arma::mat YParallel;
  const double finalObjectiveParallel = tsneParallel.Embed(X, YParallel);

  REQUIRE(finalObjectiveParallel == Approx(finalObjectiveSeq).margin(1e-12));
  REQUIRE(arma::approx_equal(YParallel, YSeq, "absdiff", 1e-12));
}

/* TSNE n_jobs does not impact output */
TEST_CASE("TSNETsneNJobsTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Run TSNE with n_jobs=1 and n_jobs=2 (angle=0, same seed) and assert
  // embeddings match.
  SUCCEED();
}

/* Numeric type coverage: double, float, etc. */
TEST_CASE("TSNETypeCoverageTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Ensure TSNE handles float32 and float64 inputs:
  //   * For projects with single-precision internals, output dtype may be
  //   float32.
  // - Assert output dtype and no crashes.
  SUCCEED();
}

/* Precomputed distances usage and validations */
TEST_CASE("TSNEPrecomputedDistancesValidationTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Validate errors are raised for:
  //   * non-square distance matrices,
  //   * non-positive distances,
  //   * sparse precomputed distances with 'exact' method.
  SUCCEED();
}

/* Chebyshev metric special-case (non-squared metric) */
TEST_CASE("TSNEChebyshevMetricTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Run TSNE with metric = "chebyshev" for a small random X and ensure no
  // exceptions.
  SUCCEED();
}

/* TSNE with Mahalanobis metric requires matrix V/VI */
TEST_CASE("TSNETsneMahalanobisTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Check that missing V/VI raises error.
  // - Build precomputed Mahalanobis distances and compare embedding to
  // metric='mahalanobis' with V provided.
  SUCCEED();
}

/* Metric coverage: Manhattan, Mahalanobis, Chebyshev */
TEST_CASE("TSNEMetricsTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Test that TSNE accepts metric="manhattan" and "mahalanobis" (requires V
  // or VI).
  // - For mahalanobis, assert that missing params raises an error and that
  // providing V
  //   reproduces a precomputed-Mahalanobis run.
  // - Check chebyshev metric runs without requiring squaring distances.
  SUCCEED();
}


/* Reduction to one component */
TEST_CASE("TSNEReductionOneComponentTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Run TSNE with n_components = 1 and check output shape (n_samples x 1)
  // and finiteness.
  SUCCEED();
}

/* Uniform grid recovery */
TEST_CASE("TSNEUniformGridRecoveryTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Use a 2D uniform grid embedded in higher-D (X_2d_grid equivalent).
  // - Run TSNE (several seeds) and check nearest-neighbor spacing: min
  // distance > 0.1,
  //   smallest/mean and largest/mean ratios in acceptable bounds.
  // - If first run fails, rerun using previous embedding as init and re-check.
  SUCCEED();
}

/* Trustworthiness tests */
TEST_CASE("TSNETrustworthinessTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Test trustworthiness for:
  //   * Affine transform -> trustworthiness == 1.0
  //   * Random permutation -> trustworthiness < 0.6
  //   * Small controlled permutation -> exact numeric trustworthiness
  // - Also implement n_neighbors validation throwing when invalid.
  SUCCEED();
}

/* Preserve trustworthiness approximately (exact/barnes_hut + init random/pca)
 */
TEST_CASE("TSNEPreserveTrustworthinessApproximatelyTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - For both methods ('exact', 'barnes_hut') and both inits ('random',
  // 'pca'):
  //   Run TSNE on random X and assert trustworthiness > 0.85 for
  //   n_neighbors=1.
  SUCCEED();
}

/** Gradient descent stopping conditions (mapping test_gradient_descent_stops)
 */
TEST_CASE("TSNEGradientDescentStopsTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - Port the ObjectiveSmallGradient and flat_function behaviours:
  //   * Check stopping on min_grad_norm triggers "gradient norm" message (if
  //   logging exists).
  //   * Check n_iter_without_progress behaviour triggers appropriate stop and
  //   message.
  //   * Check max_iter stops at expected iteration count.
  // - If your _gradient_descent writes to stdout, capture and assert messages.
  SUCCEED();
}

/* n_iter_without_progress and min_grad_norm behaviour tests */
TEST_CASE("TSNENIterWithoutProgressAndMinGradNormTests", "[TSNETest]")
{
  // IMPLEMENT:
  // - Test edge cases for n_iter_without_progress negative handling and
  // min_grad_norm.
  // - Assert the verbose output includes "did not make any progress" message
  // when appropriate.
  SUCCEED();
}

/* TSNE with various distance metrics produces consistent precomputed results
 */
TEST_CASE("TSNETsneWithDifferentDistanceMetricsTest", "[TSNETest]")
{
  // IMPLEMENT:
  // - For metric in {manhattan, cosine} and method in {barnes_hut, exact}:
  //   compute embedding with metric direct and with metric='precomputed' using
  //   the metric function.
  // - Assert the two results are equal (or raise expected xfail when known
  // mismatch).
  SUCCEED();
}
