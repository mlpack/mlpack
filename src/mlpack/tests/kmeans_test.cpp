/**
 * @file kmeans_test.cpp
 * @author Ryan Curtin
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/allow_empty_clusters.hpp>
#include <mlpack/methods/kmeans/refined_start.hpp>
#include <mlpack/methods/kmeans/elkan_kmeans.hpp>
#include <mlpack/methods/kmeans/hamerly_kmeans.hpp>
#include <mlpack/methods/kmeans/pelleg_moore_kmeans.hpp>
#include <mlpack/methods/kmeans/dual_tree_kmeans.hpp>
#include <mlpack/methods/kmeans/sample_initialization.hpp>
#include <mlpack/methods/kmeans/random_partition.hpp>

#include <mlpack/core/tree/cover_tree/cover_tree.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::kmeans;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::neighbor;

BOOST_AUTO_TEST_SUITE(KMeansTest);

// Generate dataset; written transposed because it's easier to read.
arma::mat kMeansData("  0.0   0.0;" // Class 1.
                     "  0.3   0.4;"
                     "  0.1   0.0;"
                     "  0.1   0.3;"
                     " -0.2  -0.2;"
                     " -0.1   0.3;"
                     " -0.4   0.1;"
                     "  0.2  -0.1;"
                     "  0.3   0.0;"
                     " -0.3  -0.3;"
                     "  0.1  -0.1;"
                     "  0.2  -0.3;"
                     " -0.3   0.2;"
                     " 10.0  10.0;" // Class 2.
                     " 10.1   9.9;"
                     "  9.9  10.0;"
                     " 10.2   9.7;"
                     " 10.2   9.8;"
                     "  9.7  10.3;"
                     "  9.9  10.1;"
                     "-10.0   5.0;" // Class 3.
                     " -9.8   5.1;"
                     " -9.9   4.9;"
                     "-10.0   4.9;"
                     "-10.2   5.2;"
                     "-10.1   5.1;"
                     "-10.3   5.3;"
                     "-10.0   4.8;"
                     " -9.6   5.0;"
                     " -9.8   5.1;");

/**
 * 30-point 3-class test case for K-Means.
 */
BOOST_AUTO_TEST_CASE(KMeansSimpleTest)
{
  // This test was originally written to use RandomPartition, and is left that
  // way because RandomPartition gives better initializations here.
  KMeans<EuclideanDistance, RandomPartition> kmeans;

  arma::Row<size_t> assignments;
  kmeans.Cluster((arma::mat) trans(kMeansData), 3, assignments);

  // Now make sure we got it all right.  There is no restriction on how the
  // clusters are ordered, so we have to be careful about that.
  size_t firstClass = assignments(0);

  for (size_t i = 1; i < 13; i++)
    BOOST_REQUIRE_EQUAL(assignments(i), firstClass);

  size_t secondClass = assignments(13);

  // To ensure that class 1 != class 2.
  BOOST_REQUIRE_NE(firstClass, secondClass);

  for (size_t i = 13; i < 20; i++)
    BOOST_REQUIRE_EQUAL(assignments(i), secondClass);

  size_t thirdClass = assignments(20);

  // To ensure that this is the third class which we haven't seen yet.
  BOOST_REQUIRE_NE(firstClass, thirdClass);
  BOOST_REQUIRE_NE(secondClass, thirdClass);

  for (size_t i = 20; i < 30; i++)
    BOOST_REQUIRE_EQUAL(assignments(i), thirdClass);
}

/**
 * Make sure the empty cluster policy class does nothing.
 */
BOOST_AUTO_TEST_CASE(AllowEmptyClusterTest)
{
  arma::Row<size_t> assignments;
  assignments.randu(30);
  arma::Row<size_t> assignmentsOld = assignments;

  arma::mat centroids;
  centroids.randu(30, 3); // This doesn't matter.

  arma::Col<size_t> counts(3);
  counts[0] = accu(assignments == 0);
  counts[1] = accu(assignments == 1);
  counts[2] = 0;
  arma::Col<size_t> countsOld = counts;

  // Make sure the method doesn't modify any points.
  metric::LMetric<2, true> metric;
  BOOST_REQUIRE_EQUAL(AllowEmptyClusters::EmptyCluster(kMeansData, 2, centroids,
      centroids, counts, metric, 0), 0);

  // Make sure no assignments were changed.
  for (size_t i = 0; i < assignments.n_elem; i++)
    BOOST_REQUIRE_EQUAL(assignments[i], assignmentsOld[i]);

  // Make sure no counts were changed.
  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_EQUAL(counts[i], countsOld[i]);
}

/**
 * Make sure the max variance method finds the correct point.
 */
BOOST_AUTO_TEST_CASE(MaxVarianceNewClusterTest)
{
  // Five points.
  arma::mat data("0.4 1.0 5.0 -2.0 -2.5;"
                 "1.0 0.8 0.7  5.1  5.2;");

  // Point 2 is the mis-clustered point we're looking for to be moved.
  arma::Row<size_t> assignments("0 0 0 1 1");

  arma::mat centroids(2, 3);
  centroids.col(0) = (1.0 / 3.0) * (data.col(0) + data.col(1) + data.col(2));
  centroids.col(1) = 0.5 * (data.col(3) + data.col(4));
  centroids(0, 2) = DBL_MAX;
  centroids(1, 2) = DBL_MAX;

  arma::Col<size_t> counts("3 2 0");

  metric::LMetric<2, true> metric;

  // This should only change one point.
  MaxVarianceNewCluster mvnc;
  BOOST_REQUIRE_EQUAL(mvnc.EmptyCluster(data, 2, centroids, centroids, counts,
      metric, 0), 1);

  // Add the variance of each point's distance away from the cluster.  I think
  // this is the sensible thing to do.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Find the closest centroid to this point.
    double minDistance = std::numeric_limits<double>::infinity();
    size_t closestCluster = centroids.n_cols; // Invalid value.

    for (size_t j = 0; j < centroids.n_cols; j++)
    {
      const double distance = metric.Evaluate(data.col(i), centroids.col(j));

      if (distance < minDistance)
      {
        minDistance = distance;
        closestCluster = j;
      }
    }

    assignments[i] = closestCluster;
  }

  BOOST_REQUIRE_EQUAL(assignments[0], 0);
  BOOST_REQUIRE_EQUAL(assignments[1], 0);
  BOOST_REQUIRE_EQUAL(assignments[2], 2);
  BOOST_REQUIRE_EQUAL(assignments[3], 1);
  BOOST_REQUIRE_EQUAL(assignments[4], 1);

  // Ensure that the counts are right.
  BOOST_REQUIRE_EQUAL(counts[0], 2);
  BOOST_REQUIRE_EQUAL(counts[1], 2);
  BOOST_REQUIRE_EQUAL(counts[2], 1);
}

/**
 * Make sure the random partitioner seems to return valid results.
 */
BOOST_AUTO_TEST_CASE(RandomPartitionTest)
{
  arma::mat data;
  data.randu(2, 1000); // One thousand points.

  arma::Row<size_t> assignments;

  // We'll ask for 18 clusters (arbitrary).
  RandomPartition::Cluster(data, 18, assignments);

  // Ensure that the right number of assignments were given.
  BOOST_REQUIRE_EQUAL(assignments.n_elem, 1000);

  // Ensure that no value is greater than 17 (the maximum valid cluster).
  for (size_t i = 0; i < 1000; i++)
    BOOST_REQUIRE_LT(assignments[i], 18);
}

/**
 * Make sure that random initialization fails for a corner case dataset.
 */
BOOST_AUTO_TEST_CASE(RandomInitialAssignmentFailureTest)
{
  // This is a very synthetic dataset.  It is one Gaussian with a huge number of
  // points combined with one faraway Gaussian with very few points.  Normally,
  // k-means should not get the correct result -- which is one cluster at each
  // Gaussian.  This is because the random partitioning scheme has very low
  // (virtually zero) likelihood of separating the two Gaussians properly, and
  // then the algorithm will never converge to that result.
  //
  // So we will set the initial assignments appropriately.  Remember, once the
  // initial assignments are done, k-means is deterministic.
  arma::mat dataset(2, 10002);
  dataset.randn();
  // Now move the second Gaussian far away.
  for (size_t i = 0; i < 2; ++i)
    dataset.col(10000 + i) += arma::vec("50 50");

  // Ensure that k-means fails when run with random initialization.  This isn't
  // strictly a necessary test, but it does help let us know that this is a good
  // test.
  size_t successes = 0;
  for (size_t run = 0; run < 15; ++run)
  {
    arma::mat centroids;
    arma::Row<size_t> assignments;
    KMeans<> kmeans;
    kmeans.Cluster(dataset, 2, assignments, centroids);

    // Inspect centroids.  See if one is close to the second Gaussian.
    if ((centroids(0, 0) >= 30.0 && centroids(1, 0) >= 30.0) ||
        (centroids(0, 1) >= 30.0 && centroids(1, 1) >= 30.0))
      ++successes;
  }

  // Only one success allowed.  The probability of two successes should be
  // infinitesimal.
  BOOST_REQUIRE_LT(successes, 2);
}

/**
 * Make sure that specifying initial assignments is successful for a corner case
 * dataset which doesn't usually converge otherwise.
 */
BOOST_AUTO_TEST_CASE(InitialAssignmentTest)
{
  // For a better description of this dataset, see
  // RandomInitialAssignmentFailureTest.
  arma::mat dataset(2, 10002);
  dataset.randn();
  // Now move the second Gaussian far away.
  for (size_t i = 0; i < 2; ++i)
    dataset.col(10000 + i) += arma::vec("50 50");

  // Now, if we specify initial assignments, the algorithm should converge (with
  // zero iterations, actually, because this is the solution).
  arma::Row<size_t> assignments(10002);
  assignments.fill(0);
  assignments[10000] = 1;
  assignments[10001] = 1;

  KMeans<> kmeans;
  kmeans.Cluster(dataset, 2, assignments, true);

  // Check results.
  for (size_t i = 0; i < 10000; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 0);
  for (size_t i = 10000; i < 10002; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 1);

  // Now, slightly harder.  Give it one incorrect assignment in each cluster.
  // The wrong assignment should be quickly fixed.
  assignments[9999] = 1;
  assignments[10000] = 0;

  kmeans.Cluster(dataset, 2, assignments, true);

  // Check results.
  for (size_t i = 0; i < 10000; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 0);
  for (size_t i = 10000; i < 10002; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 1);
}

/**
 * Make sure specifying initial centroids is successful for a corner case which
 * doesn't usually converge otherwise.
 */
BOOST_AUTO_TEST_CASE(InitialCentroidTest)
{
  // For a better description of this dataset, see
  // RandomInitialAssignmentFailureTest.
  arma::mat dataset(2, 10002);
  dataset.randn();
  // Now move the second Gaussian far away.
  for (size_t i = 0; i < 2; ++i)
    dataset.col(10000 + i) += arma::vec("50 50");

  arma::Row<size_t> assignments;
  arma::mat centroids(2, 2);

  centroids.col(0) = arma::vec("0 0");
  centroids.col(1) = arma::vec("50 50");

  // This should converge correctly.
  KMeans<> k;
  k.Cluster(dataset, 2, assignments, centroids, false, true);

  // Check results.
  for (size_t i = 0; i < 10000; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 0);
  for (size_t i = 10000; i < 10002; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 1);

  // Now add a little noise to the initial centroids.
  centroids.col(0) = arma::vec("3 4");
  centroids.col(1) = arma::vec("25 10");

  k.Cluster(dataset, 2, assignments, centroids, false, true);

  // Check results.
  for (size_t i = 0; i < 10000; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 0);
  for (size_t i = 10000; i < 10002; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 1);
}

/**
 * Ensure that initial assignments override initial centroids.
 */
BOOST_AUTO_TEST_CASE(InitialAssignmentOverrideTest)
{
  // For a better description of this dataset, see
  // RandomInitialAssignmentFailureTest.
  arma::mat dataset(2, 10002);
  dataset.randn();
  // Now move the second Gaussian far away.
  for (size_t i = 0; i < 2; ++i)
    dataset.col(10000 + i) += arma::vec("50 50");

  arma::Row<size_t> assignments(10002);
  assignments.fill(0);
  assignments[10000] = 1;
  assignments[10001] = 1;

  // Note that this initial centroid guess is the opposite of the assignments
  // guess!
  arma::mat centroids(2, 2);
  centroids.col(0) = arma::vec("50 50");
  centroids.col(1) = arma::vec("0 0");

  KMeans<> k;
  k.Cluster(dataset, 2, assignments, centroids, true, true);

  // Because the initial assignments guess should take priority, we should get
  // those same results back.
  for (size_t i = 0; i < 10000; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 0);
  for (size_t i = 10000; i < 10002; ++i)
    BOOST_REQUIRE_EQUAL(assignments[i], 1);

  // Make sure the centroids are about right too.
  BOOST_REQUIRE_LT(centroids(0, 0), 10.0);
  BOOST_REQUIRE_LT(centroids(1, 0), 10.0);
  BOOST_REQUIRE_GT(centroids(0, 1), 40.0);
  BOOST_REQUIRE_GT(centroids(1, 1), 40.0);
}

/**
 * Test that the refined starting policy returns decent initial cluster
 * estimates.
 */
BOOST_AUTO_TEST_CASE(RefinedStartTest)
{
  // Our dataset will be five Gaussians of largely varying numbers of points and
  // we expect that the refined starting policy should return good guesses at
  // what these Gaussians are.
  arma::mat data(3, 3000);
  data.randn();

  // First Gaussian: 10000 points, centered at (0, 0, 0).
  // Second Gaussian: 2000 points, centered at (5, 0, -2).
  // Third Gaussian: 5000 points, centered at (-2, -2, -2).
  // Fourth Gaussian: 1000 points, centered at (-6, 8, 8).
  // Fifth Gaussian: 12000 points, centered at (1, 6, 1).
  arma::mat centroids(" 0  5 -2 -6  1;"
                      " 0  0 -2  8  6;"
                      " 0 -2 -2  8  1");

  for (size_t i = 1000; i < 1200; ++i)
    data.col(i) += centroids.col(1);
  for (size_t i = 1200; i < 1700; ++i)
    data.col(i) += centroids.col(2);
  for (size_t i = 1700; i < 1800; ++i)
    data.col(i) += centroids.col(3);
  for (size_t i = 1800; i < 3000; ++i)
    data.col(i) += centroids.col(4);

  // Now run the RefinedStart algorithm and make sure it doesn't deviate too
  // much from the actual solution.
  RefinedStart rs;
  arma::Row<size_t> assignments;
  arma::mat resultingCentroids;
  rs.Cluster(data, 5, assignments);

  // Calculate resulting centroids.
  resultingCentroids.zeros(3, 5);
  arma::Col<size_t> counts(5);
  counts.zeros();
  for (size_t i = 0; i < 3000; ++i)
  {
    resultingCentroids.col(assignments[i]) += data.col(i);
    ++counts[assignments[i]];
  }

  // Normalize centroids.
  for (size_t i = 0; i < 5; ++i)
    if (counts[i] != 0)
      resultingCentroids /= counts[i];

  // Calculate sum of distances from centroid means.
  double distortion = 0;
  for (size_t i = 0; i < 3000; ++i)
    distortion += metric::EuclideanDistance::Evaluate(data.col(i),
        resultingCentroids.col(assignments[i]));

  // Using the refined start, the distance for this dataset is usually around
  // 13500.  Regular k-means is between 10000 and 30000 (I think the 10000
  // figure is a corner case which actually does not give good clusters), and
  // random initial starts give distortion around 22000.  So we'll require that
  // our distortion is less than 14000.
  BOOST_REQUIRE_LT(distortion, 14000.0);
}

#ifdef ARMA_HAS_SPMAT
// Can't do this test on Armadillo 3.4; var(SpBase) is not implemented.
#if !((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR == 4))

/**
 * Make sure sparse k-means works okay.
 */
BOOST_AUTO_TEST_CASE(SparseKMeansTest)
{
  // Huge dimensionality, few points.
  arma::SpMat<double> data(5000, 12);
  data(14, 0) = 6.4;
  data(14, 1) = 6.3;
  data(14, 2) = 6.5;
  data(14, 3) = 6.2;
  data(14, 4) = 6.1;
  data(14, 5) = 6.6;
  data(1402, 6) = -3.2;
  data(1402, 7) = -3.3;
  data(1402, 8) = -3.1;
  data(1402, 9) = -3.4;
  data(1402, 10) = -3.5;
  data(1402, 11) = -3.0;

  arma::Row<size_t> assignments;

  KMeans<metric::EuclideanDistance, RandomPartition, MaxVarianceNewCluster,
         NaiveKMeans, arma::sp_mat> kmeans; // Default options.

  kmeans.Cluster(data, 2, assignments);

  size_t clusterOne = assignments[0];
  size_t clusterTwo = assignments[6];

  BOOST_REQUIRE_EQUAL(assignments[0], clusterOne);
  BOOST_REQUIRE_EQUAL(assignments[1], clusterOne);
  BOOST_REQUIRE_EQUAL(assignments[2], clusterOne);
  BOOST_REQUIRE_EQUAL(assignments[3], clusterOne);
  BOOST_REQUIRE_EQUAL(assignments[4], clusterOne);
  BOOST_REQUIRE_EQUAL(assignments[5], clusterOne);
  BOOST_REQUIRE_EQUAL(assignments[6], clusterTwo);
  BOOST_REQUIRE_EQUAL(assignments[7], clusterTwo);
  BOOST_REQUIRE_EQUAL(assignments[8], clusterTwo);
  BOOST_REQUIRE_EQUAL(assignments[9], clusterTwo);
  BOOST_REQUIRE_EQUAL(assignments[10], clusterTwo);
  BOOST_REQUIRE_EQUAL(assignments[11], clusterTwo);
}

#endif // Exclude Armadillo 3.4.
#endif // ARMA_HAS_SPMAT

BOOST_AUTO_TEST_CASE(ElkanTest)
{
  const size_t trials = 5;

  for (size_t t = 0; t < trials; ++t)
  {
    arma::mat dataset(10, 1000);
    dataset.randu();

    const size_t k = 5 * (t + 1);
    arma::mat centroids(10, k);
    centroids.randu();

    // Make sure Elkan's algorithm and the naive method return the same
    // clusters.
    arma::mat naiveCentroids(centroids);
    KMeans<> km;
    arma::Row<size_t> assignments;
    km.Cluster(dataset, k, assignments, naiveCentroids, false, true);

    KMeans<metric::EuclideanDistance, RandomPartition, MaxVarianceNewCluster,
         ElkanKMeans> elkan;
    arma::Row<size_t> elkanAssignments;
    arma::mat elkanCentroids(centroids);
    elkan.Cluster(dataset, k, elkanAssignments, elkanCentroids, false, true);

    for (size_t i = 0; i < dataset.n_cols; ++i)
      BOOST_REQUIRE_EQUAL(assignments[i], elkanAssignments[i]);

    for (size_t i = 0; i < centroids.n_elem; ++i)
      BOOST_REQUIRE_CLOSE(naiveCentroids[i], elkanCentroids[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(HamerlyTest)
{
  const size_t trials = 5;

  for (size_t t = 0; t < trials; ++t)
  {
    arma::mat dataset(10, 1000);
    dataset.randu();

    const size_t k = 5 * (t + 1);
    arma::mat centroids(10, k);
    centroids.randu();

    // Make sure Hamerly's algorithm and the naive method return the same
    // clusters.
    arma::mat naiveCentroids(centroids);
    KMeans<> km;
    arma::Row<size_t> assignments;
    km.Cluster(dataset, k, assignments, naiveCentroids, false, true);

    KMeans<metric::EuclideanDistance, RandomPartition, MaxVarianceNewCluster,
        HamerlyKMeans> hamerly;
    arma::Row<size_t> hamerlyAssignments;
    arma::mat hamerlyCentroids(centroids);
    hamerly.Cluster(dataset, k, hamerlyAssignments, hamerlyCentroids, false,
        true);

    for (size_t i = 0; i < dataset.n_cols; ++i)
      BOOST_REQUIRE_EQUAL(assignments[i], hamerlyAssignments[i]);

    for (size_t i = 0; i < centroids.n_elem; ++i)
      BOOST_REQUIRE_CLOSE(naiveCentroids[i], hamerlyCentroids[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(PellegMooreTest)
{
  const size_t trials = 5;

  for (size_t t = 0; t < trials; ++t)
  {
    arma::mat dataset(10, 1000);
    dataset.randu();

    const size_t k = 5 * (t + 1);
    arma::mat centroids(10, k);
    centroids.randu();

    // Make sure the Pelleg-Moore algorithm and the naive method return the same
    // clusters.
    arma::mat naiveCentroids(centroids);
    KMeans<> km;
    arma::Row<size_t> assignments;
    km.Cluster(dataset, k, assignments, naiveCentroids, false, true);

    KMeans<metric::EuclideanDistance, RandomPartition, MaxVarianceNewCluster,
        PellegMooreKMeans> pellegMoore;
    arma::Row<size_t> pmAssignments;
    arma::mat pmCentroids(centroids);
    pellegMoore.Cluster(dataset, k, pmAssignments, pmCentroids, false, true);

    for (size_t i = 0; i < dataset.n_cols; ++i)
      BOOST_REQUIRE_EQUAL(assignments[i], pmAssignments[i]);

    for (size_t i = 0; i < centroids.n_elem; ++i)
      BOOST_REQUIRE_CLOSE(naiveCentroids[i], pmCentroids[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(DTNNTest)
{
  const size_t trials = 5;

  for (size_t t = 0; t < trials; ++t)
  {
    arma::mat dataset(10, 1000);
    dataset.randu();

    const size_t k = 5 * (t + 1);
    arma::mat centroids(10, k);
    centroids.randu();

    arma::mat naiveCentroids(centroids);
    KMeans<> km;
    arma::Row<size_t> assignments;
    km.Cluster(dataset, k, assignments, naiveCentroids, false, true);

    KMeans<metric::EuclideanDistance, RandomPartition, MaxVarianceNewCluster,
        DefaultDualTreeKMeans> dtnn;
    arma::Row<size_t> dtnnAssignments;
    arma::mat dtnnCentroids(centroids);
    dtnn.Cluster(dataset, k, dtnnAssignments, dtnnCentroids, false, true);

    for (size_t i = 0; i < dataset.n_cols; ++i)
      BOOST_REQUIRE_EQUAL(assignments[i], dtnnAssignments[i]);

    for (size_t i = 0; i < centroids.n_elem; ++i)
      BOOST_REQUIRE_CLOSE(naiveCentroids[i], dtnnCentroids[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(DTNNCoverTreeTest)
{
  const size_t trials = 5;

  for (size_t t = 0; t < trials; ++t)
  {
    arma::mat dataset(10, 1000);
    dataset.randu();

    const size_t k = 5;
    arma::mat centroids(10, k);
    centroids.randu();

    arma::mat naiveCentroids(centroids);
    KMeans<> km;
    arma::Row<size_t> assignments;
    km.Cluster(dataset, k, assignments, naiveCentroids, false, true);

    KMeans<metric::EuclideanDistance, RandomPartition, MaxVarianceNewCluster,
        CoverTreeDualTreeKMeans> dtnn;
    arma::Row<size_t> dtnnAssignments;
    arma::mat dtnnCentroids(centroids);
    dtnn.Cluster(dataset, k, dtnnAssignments, dtnnCentroids, false, true);

    for (size_t i = 0; i < dataset.n_cols; ++i)
      BOOST_REQUIRE_EQUAL(assignments[i], dtnnAssignments[i]);

    for (size_t i = 0; i < centroids.n_elem; ++i)
      BOOST_REQUIRE_CLOSE(naiveCentroids[i], dtnnCentroids[i], 1e-5);
  }
}

/**
 * Make sure that the sample initialization strategy successfully samples points
 * from the dataset.
 */
BOOST_AUTO_TEST_CASE(SampleInitializationTest)
{
  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  const size_t clusters = 10;
  arma::mat centroids;

  SampleInitialization::Cluster(dataset, clusters, centroids);

  // Check that the size of the matrix is correct.
  BOOST_REQUIRE_EQUAL(centroids.n_cols, 10);
  BOOST_REQUIRE_EQUAL(centroids.n_rows, 5);

  // Check that each entry in the matrix is some sample from the dataset.
  for (size_t i = 0; i < clusters; ++i)
  {
    // If the loop successfully terminates, j will be equal to dataset.n_cols.
    // If not then we have found a match.
    size_t j;
    for (j = 0; j < dataset.n_cols; ++j)
    {
      const double distance = metric::EuclideanDistance::Evaluate(
          centroids.col(i), dataset.col(j));
      if (distance < 1e-10)
        break;
    }

    BOOST_REQUIRE_LT(j, dataset.n_cols);
  }
}

BOOST_AUTO_TEST_SUITE_END();
