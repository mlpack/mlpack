/**
 * @file mean_shift_test.cpp
 * @author Shangtong Zhang
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/mean_shift/mean_shift.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::meanshift;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(MeanShiftTest);

// Generate dataset; written transposed because it's easier to read.
arma::mat meanShiftData("  0.0   0.0;" // Class 1.
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
 * 30-point 3-class test case for Mean Shift.
 */
BOOST_AUTO_TEST_CASE(MeanShiftSimpleTest) {

  MeanShift<> meanShift;

  arma::Col<size_t> assignments;
  arma::mat centroids;
  meanShift.Cluster((arma::mat) trans(meanShiftData), assignments, centroids);

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

// Generate samples from four Gaussians, and make sure mean shift nearly
// recovers those four centers.
BOOST_AUTO_TEST_CASE(GaussianClustering)
{
  GaussianDistribution g1("0.0 0.0 0.0", arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2("5.0 5.0 5.0", 2 * arma::eye<arma::mat>(3, 3));
  GaussianDistribution g3("-3.0 3.0 -1.0", arma::eye<arma::mat>(3, 3));
  GaussianDistribution g4("6.0 -2.0 -2.0", 3 * arma::eye<arma::mat>(3, 3));

  arma::mat dataset(3, 4000);
  for (size_t i = 0; i < 1000; ++i)
    dataset.col(i) = g1.Random();
  for (size_t i = 1000; i < 2000; ++i)
    dataset.col(i) = g2.Random();
  for (size_t i = 2000; i < 3000; ++i)
    dataset.col(i) = g3.Random();
  for (size_t i = 3000; i < 4000; ++i)
    dataset.col(i) = g4.Random();

  // Now that the dataset is generated, run mean shift.  Pre-set radius.
  MeanShift<> meanShift(2.9);

  arma::Col<size_t> assignments;
  arma::mat centroids;
  meanShift.Cluster(dataset, assignments, centroids);

  BOOST_REQUIRE_EQUAL(centroids.n_cols, 4);
  BOOST_REQUIRE_EQUAL(centroids.n_rows, 3);

  std::cout << centroids.t();

  // Check that each centroid is close to only one mean.
  arma::vec centroidDistances(4);
  arma::uvec minIndices(4);
  for (size_t i = 0; i < 4; ++i)
  {
    centroidDistances(0) = metric::EuclideanDistance::Evaluate(g1.Mean(),
        centroids.col(i));
    centroidDistances(1) = metric::EuclideanDistance::Evaluate(g2.Mean(),
        centroids.col(i));
    centroidDistances(2) = metric::EuclideanDistance::Evaluate(g3.Mean(),
        centroids.col(i));
    centroidDistances(3) = metric::EuclideanDistance::Evaluate(g4.Mean(),
        centroids.col(i));

    // Are we near a centroid of a Gaussian?
    const double minVal = centroidDistances.min(minIndices[i]);
    BOOST_REQUIRE_SMALL(minVal, 0.65); // A decent amount of tolerance...
  }

  // Ensure each centroid corresponds to a different Gaussian.
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = i + 1; j < 4; ++j)
      BOOST_REQUIRE_NE(minIndices[i], minIndices[j]);
}

BOOST_AUTO_TEST_SUITE_END();
