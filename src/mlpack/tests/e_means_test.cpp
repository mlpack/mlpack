/**
 * @file tests/e_means_test.cpp
 *
 * Tests for EMeans.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;

// Sum of squared distances from each point to its assigned centroid.
template<typename MatType>
static double Inertia(const MatType& data,
                      const MatType& centroids,
                      const arma::Row<size_t>& assignments)
{
  double total = 0.0;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    total += arma::accu(arma::square(arma::conv_to<arma::vec>::from(
        data.col(i) - centroids.col(assignments[i]))));
  }
  return total;
}

// Three well-separated Gaussians.
static arma::mat ThreeBlobs(const size_t perBlob, const size_t d)
{
  arma::mat data(d, 3 * perBlob);
  data.cols(0, perBlob - 1) = arma::randn<arma::mat>(d, perBlob);
  data.cols(perBlob, 2 * perBlob - 1) =
      arma::randn<arma::mat>(d, perBlob) + 20.0;
  data.cols(2 * perBlob, 3 * perBlob - 1) =
      arma::randn<arma::mat>(d, perBlob) - 20.0;
  return data;
}

/**
 * e-means separates three well-separated blobs from given initial centroids.
 */
TEST_CASE("EMeansThreeBlobsTest", "[EMeansTest]")
{
  arma::mat data = ThreeBlobs(500, 5);
  arma::mat centroids = data.cols(arma::uvec({ 0, 500, 1000 }));

  EMeans<> em;
  arma::Row<size_t> assignments;
  em.Cluster(data, 3, assignments, centroids, true);

  REQUIRE(centroids.n_rows == 5);
  REQUIRE(centroids.n_cols == 3);
  for (size_t b = 0; b < 3; ++b)
  {
    const size_t label = assignments[500 * b];
    REQUIRE(arma::all(assignments.subvec(500 * b, 500 * b + 499) == label));
  }
  REQUIRE(em.Steps() > 5);
  REQUIRE(em.UsedBatchSize() > 0);
  REQUIRE(em.UsedAlpha0() > 0.0);
  REQUIRE(em.UsedAlpha0() <= 1.0);
}

/**
 * With a full batch, a step of 1 and no revival, each epoch is one Lloyd
 * iteration.
 */
TEST_CASE("EMeansFullBatchIsLloydTest", "[EMeansTest]")
{
  arma::mat data = arma::randu<arma::mat>(4, 600);
  arma::mat init = data.cols(0, 9);

  arma::mat emCentroids = init;
  EMeans<> em(1.0, data.n_cols, 1.0, EMeans<>::AUTO, 0.0);
  em.Cluster(data, 10, emCentroids, true);

  arma::mat kmCentroids = init;
  KMeans<> km(1);
  km.Cluster(data, 10, kmCentroids, true);

  REQUIRE(arma::approx_equal(emCentroids, kmCentroids, "absdiff", 1e-10));
}

/**
 * e-means reaches about the error of 25 Lloyd iterations in 5 epochs.
 */
TEST_CASE("EMeansCompetitiveWithLloydTest", "[EMeansTest]")
{
  arma::mat data = arma::randn<arma::mat>(32, 20000);
  arma::mat init = data.cols(arma::randperm(data.n_cols, 100));

  arma::mat emCentroids = init;
  arma::Row<size_t> emAssignments;
  EMeans<> em;
  em.Cluster(data, 100, emAssignments, emCentroids, true);

  arma::mat kmCentroids = init;
  arma::Row<size_t> kmAssignments;
  KMeans<> km(25);
  km.Cluster(data, 100, kmAssignments, kmCentroids, false, true);

  const double emInertia = Inertia(data, emCentroids, emAssignments);
  const double kmInertia = Inertia(data, kmCentroids, kmAssignments);
  REQUIRE(emInertia < 1.01 * kmInertia);
}

/**
 * Assignments are to the closest centroid.
 */
TEST_CASE("EMeansAssignTest", "[EMeansTest]")
{
  arma::mat data = arma::randn<arma::mat>(7, 300);
  arma::mat centroids = arma::randn<arma::mat>(7, 13);
  arma::Row<size_t> assignments;
  arma::vec distances;
  EMeans<>::Assign(data, centroids, assignments, &distances);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::vec d(13);
    for (size_t c = 0; c < 13; ++c)
      d[c] = arma::accu(arma::square(data.col(i) - centroids.col(c)));
    REQUIRE(assignments[i] == d.index_min());
    REQUIRE(distances[i] == Approx(d.min()).epsilon(1e-8));
  }
}

/**
 * Dead clusters are revived by splits, and are not if revival is disabled.
 */
TEST_CASE("EMeansRevivalTest", "[EMeansTest]")
{
  arma::mat data = arma::randn<arma::mat>(8, 4000);
  // All points start closest to the first centroid.
  arma::mat init = 1000.0 + arma::randn<arma::mat>(8, 20);
  init.col(0).zeros();

  arma::mat centroids = init;
  arma::Row<size_t> assignments;
  EMeans<> em;
  em.Cluster(data, 20, assignments, centroids, true);
  REQUIRE(em.Splits() > 0);
  REQUIRE(arma::unique(assignments).eval().n_elem == 20);

  centroids = init;
  EMeans<> noRevival(5.0, 0, 0.0, EMeans<>::AUTO, 0.0);
  noRevival.Cluster(data, 20, assignments, centroids, true);
  REQUIRE(noRevival.Splits() == 0);
  REQUIRE(arma::unique(assignments).eval().n_elem < 20);
}

/**
 * Manual batch size and step are used as given.
 */
TEST_CASE("EMeansManualBatchTest", "[EMeansTest]")
{
  arma::mat data = ThreeBlobs(100, 3);
  arma::mat centroids;
  EMeans<> em(5.0, 30, 0.3);
  em.Cluster(data, 3, centroids);
  REQUIRE(em.UsedBatchSize() == 30);
  REQUIRE(em.UsedAlpha0() == 0.3);
  REQUIRE(em.Steps() == 5 * 10);
}

/**
 * Fractional numbers of epochs.
 */
TEST_CASE("EMeansEpochRowsTest", "[EMeansTest]")
{
  REQUIRE(EMeans<>::EpochRows(1000, 1.5) == std::vector<size_t>({ 500, 1000 }));
  REQUIRE(EMeans<>::EpochRows(1000, 0.5) == std::vector<size_t>({ 500 }));
  REQUIRE(EMeans<>::EpochRows(1000, 3.0) ==
      std::vector<size_t>({ 1000, 1000, 1000 }));
}

/**
 * The scaling laws and the step schedule.
 */
TEST_CASE("EMeansLawsTest", "[EMeansTest]")
{
  const size_t b = EMeans<>::BatchSizeLaw(1000000, 4000, 5.0, 128);
  REQUIRE((b & (b - 1)) == 0); // Power of two.
  REQUIRE(EMeans<>::BatchSizeLaw(1000000, 4000, 5.0, 8) <= b / 4);
  REQUIRE(EMeans<>::BatchSizeLaw(10, 4, 5.0, 128) <= 10);
  const double a = EMeans<>::StepLaw(1000000, 4000, 5.0, b);
  REQUIRE(a > 0.0);
  REQUIRE(a <= 1.0);
  REQUIRE(EMeans<>::ScheduledAlpha(0.5, 0, 10) == Approx(0.5));
  REQUIRE(EMeans<>::ScheduledAlpha(0.5, 10, 10) == Approx(0.05));
}

/**
 * Single-precision data.
 */
TEST_CASE("EMeansFloatTest", "[EMeansTest]")
{
  arma::fmat data = arma::conv_to<arma::fmat>::from(ThreeBlobs(300, 4));
  arma::fmat centroids;
  arma::Row<size_t> assignments;
  EMeans<> em;
  em.Cluster(data, 3, assignments, centroids);
  REQUIRE(centroids.n_rows == 4);
  REQUIRE(centroids.n_cols == 3);
  REQUIRE(assignments.n_elem == 900);
}

/**
 * Invalid arguments throw.
 */
TEST_CASE("EMeansInvalidTest", "[EMeansTest]")
{
  arma::mat data = arma::randu<arma::mat>(3, 10);
  arma::mat centroids;
  EMeans<> em;
  REQUIRE_THROWS_AS(em.Cluster(data, 11, centroids), std::invalid_argument);
  REQUIRE_THROWS_AS(em.Cluster(data, 0, centroids), std::invalid_argument);
  EMeans<> badEpochs(0.0);
  REQUIRE_THROWS_AS(badEpochs.Cluster(data, 2, centroids),
      std::invalid_argument);
  EMeans<> badAlpha(5.0, 0, 1.5);
  REQUIRE_THROWS_AS(badAlpha.Cluster(data, 2, centroids),
      std::invalid_argument);
  EMeans<> infiniteEpochs(std::numeric_limits<double>::infinity());
  REQUIRE_THROWS_AS(infiniteEpochs.Cluster(data, 2, centroids),
      std::invalid_argument);

  // The static interfaces check their arguments too.
  arma::Row<size_t> assignments;
  arma::mat noCentroids(3, 0);
  REQUIRE_THROWS_AS(EMeans<>::Assign(data, noCentroids, assignments),
      std::invalid_argument);
  arma::mat wrongDimension(2, 2, arma::fill::zeros);
  REQUIRE_THROWS_AS(EMeans<>::Assign(data, wrongDimension, assignments),
      std::invalid_argument);
  centroids = data.cols(0, 1);
  size_t steps, splits;
  const auto order = [](const size_t) { return arma::regspace<arma::uvec>(
      0, 9); };
  const auto uniform = [](const size_t k) { return arma::randu<arma::vec>(
      k); };
  REQUIRE_THROWS_AS(EMeans<>::Train(data, centroids, 0, 0.5, 1.0, 0.0, order,
      uniform, steps, splits), std::invalid_argument);
}

/**
 * Serialization keeps the parameters.
 */
TEST_CASE("EMeansSerializationTest", "[EMeansTest]")
{
  EMeans<> em(2.5, 64, 0.2, EMeans<>::QUALITY, 0.3);
  EMeans<> xmlEm, jsonEm, binaryEm;
  SerializeObjectAll(em, xmlEm, jsonEm, binaryEm);
  for (EMeans<>* e : { &xmlEm, &jsonEm, &binaryEm })
  {
    REQUIRE(e->Epochs() == 2.5);
    REQUIRE(e->BatchSize() == 64);
    REQUIRE(e->Alpha0() == 0.2);
    REQUIRE(e->Law() == EMeans<>::QUALITY);
    REQUIRE(e->UnusedThreshold() == 0.3);
  }
}
