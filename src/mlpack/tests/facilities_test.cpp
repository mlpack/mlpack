/**
 * @file facilities_test.cpp
 * @author Khizir Siddiqui
 *
 * Test file for facilities in metrics.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include "catch.hpp"

using namespace mlpack;

/**
 * Pairwise distances.
 */
TEST_CASE("PairwiseDistanceTest", "[FacilitiesTest]")
{
  arma::mat X;
  X = { { 0, 1, 1, 0, 0 },
        { 0, 1, 2, 0, 0 },
        { 1, 1, 3, 2, 0 } };
  EuclideanDistance metric;
  
  // Compute the pairwise distance matrix
  arma::mat dist = PairwiseDistances(X, metric);
  
  // Testing distance from each point to itself (should be 0)
  REQUIRE(dist(0, 0) == 0);
  REQUIRE(dist(1, 1) == 0);
  REQUIRE(dist(2, 2) == 0);
  
  // Testing known distances
  REQUIRE(dist(0, 1) == Approx(1.41421).epsilon(1e-5)); // Euclidean distance between points 0 and 1
  REQUIRE(dist(0, 2) == 3); // Euclidean distance between points 0 and 2
  REQUIRE(dist(1, 0) == Approx(1.41421).epsilon(1e-5)); // Check symmetry: dist(0, 1) == dist(1, 0)
  REQUIRE(dist(2, 0) == 3); // Check symmetry: dist(0, 2) == dist(2, 0)
  
  // Ensure symmetry of the distance matrix
  for (size_t i = 0; i < dist.n_cols; ++i)
  {
    for (size_t j = 0; j < dist.n_rows; ++j)
    {
      REQUIRE(dist(i, j) == Approx(dist(j, i)).epsilon(1e-5));
    }
  }
  
  // Edge case: Single point (distance to itself should be zero)
  arma::mat singlePoint = { { 0, 0 } };
  arma::mat singlePointDist = PairwiseDistances(singlePoint, metric);
  REQUIRE(singlePointDist(0, 0) == 0);

  // Edge case: All points are the same (distance between any pair should be zero)
  arma::mat identicalPoints = { { 1, 1 }, { 1, 1 }, { 1, 1 } };
  arma::mat identicalDist = PairwiseDistances(identicalPoints, metric);
  for (size_t i = 0; i < identicalDist.n_cols; ++i)
  {
    for (size_t j = 0; j < identicalDist.n_rows; ++j)
    {
      REQUIRE(identicalDist(i, j) == 0);
    }
  }

  // Test with zero vector (distance to zero vector should be the norm of the other point)
  arma::mat zeroVector = { { 0, 0 }, { 0, 0 } }; // Two points: one is a zero vector
  arma::mat zeroVecDist = PairwiseDistances(zeroVector, metric);
  REQUIRE(zeroVecDist(0, 1) == Approx(0).epsilon(1e-5)); // Zero vector to zero vector
  REQUIRE(zeroVecDist(1, 0) == Approx(0).epsilon(1e-5)); // Zero vector to zero vector
  
  // Test for distance between points with negative coordinates
  arma::mat negativeCoords = { { -1, -2 }, { -3, -4 }, { -5, -6 } };
  arma::mat negativeDist = PairwiseDistances(negativeCoords, metric);
  REQUIRE(negativeDist(0, 1) == Approx(2.82843).epsilon(1e-5)); // Distance between (-1,-2) and (-3,-4)
  REQUIRE(negativeDist(1, 2) == Approx(2.82843).epsilon(1e-5)); // Distance between (-3,-4) and (-5,-6)

  // Checking the pairwise distance matrix fully
  REQUIRE(dist(0, 1) == Approx(1.41421).epsilon(1e-5));
  REQUIRE(dist(0, 2) == 3);
  REQUIRE(dist(1, 0) == Approx(1.41421).epsilon(1e-5));
  REQUIRE(dist(1, 2) == Approx(2.23607).epsilon(1e-5));
  REQUIRE(dist(2, 0) == 3);
  REQUIRE(dist(2, 1) == Approx(2.23607).epsilon(1e-5));
}
