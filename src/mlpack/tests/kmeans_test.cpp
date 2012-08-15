/**
 * @file kmeans_test.cpp
 * @author Ryan Curtin
 *
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/allow_empty_clusters.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::kmeans;

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
 * 30-point 3-class test case for K-Means, with no overclustering.
 */
BOOST_AUTO_TEST_CASE(KMeansNoOverclusteringTest)
{
  KMeans<> kmeans; // No overclustering.

  arma::Col<size_t> assignments;
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
 * 30-point 3-class test case for K-Means, with overclustering.
 */
BOOST_AUTO_TEST_CASE(KMeansOverclusteringTest)
{
  KMeans<> kmeans(1000, 4.0); // Overclustering factor of 4.0.

  arma::Col<size_t> assignments;
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
  arma::Col<size_t> assignments;
  assignments.randu(30);
  arma::Col<size_t> assignmentsOld = assignments;

  arma::mat centroids;
  centroids.randu(30, 3); // This doesn't matter.

  arma::Col<size_t> counts(3);
  counts[0] = accu(assignments == 0);
  counts[1] = accu(assignments == 1);
  counts[2] = 0;
  arma::Col<size_t> countsOld = counts;

  // Make sure the method doesn't modify any points.
  BOOST_REQUIRE_EQUAL(AllowEmptyClusters::EmptyCluster(kMeansData, 2, centroids,
      counts, assignments), 0);

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
  arma::Col<size_t> assignments("0 0 0 1 1");

  arma::mat centroids(2, 3);
  centroids.col(0) = (1.0 / 3.0) * (data.col(0) + data.col(1) + data.col(2));
  centroids.col(1) = 0.5 * (data.col(3) + data.col(4));
  centroids(0, 2) = 0;
  centroids(1, 2) = 0;

  arma::Col<size_t> counts("3 2 0");

  // This should only change one point.
  BOOST_REQUIRE_EQUAL(MaxVarianceNewCluster::EmptyCluster(data, 2, centroids,
      counts, assignments), 1);

  // Ensure that the cluster assignments are right.
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

  arma::Col<size_t> assignments;

  // We'll ask for 18 clusters (arbitrary).
  RandomPartition::Cluster(data, 18, assignments);

  // Ensure that the right number of assignments were given.
  BOOST_REQUIRE_EQUAL(assignments.n_elem, 1000);

  // Ensure that no value is greater than 17 (the maximum valid cluster).
  for (size_t i = 0; i < 1000; i++)
    BOOST_REQUIRE_LT(assignments[i], 18);
}

/**
 * Make sure sparse k-means works okay.
 *
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

  arma::Col<size_t> assignments;

  KMeans<> kmeans; // Default options.

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
*/

BOOST_AUTO_TEST_SUITE_END();
