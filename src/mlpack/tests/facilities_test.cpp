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
  arma::mat dist = PairwiseDistances(X, metric);
  REQUIRE(dist(0, 0) == 0);
  REQUIRE(dist(1, 0) == Approx(1.41421).epsilon(1e-5));
  REQUIRE(dist(2, 0) == 3);
}
