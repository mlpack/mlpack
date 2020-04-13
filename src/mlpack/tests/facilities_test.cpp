/**
 * @file facilities_test.cpp
 * @author Khizir Siddiqui
 * @author Bisakh Mondal
 * 
 * Test file for Utility facilities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/data/load.hpp>
#include <boost/test/unit_test.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::cv;
using namespace mlpack::util;

BOOST_AUTO_TEST_SUITE(FacilityTest);

/**
 * The unequal sizes for data and labels show throw an error.
 */
TEST_CASE("AssertSizesTest", "[FacilitiesTest]")
{
  // Load the dataset.
  arma::mat dataset;
  if (!data::Load("iris_train.csv", dataset))
    FAIL("Cannot load test dataset iris_train.csv!");
  // Load the labels.
  arma::Row<size_t> labels;
  if (!data::Load("iris_test_labels.csv", labels))
    FAIL("Cannot load test dataset iris_test_labels.csv!");

  REQUIRE_THROWS_AS(
    AssertSizes(dataset, labels, "test"), std::invalid_argument);
}


/**
 * Pairwise distances.
 */
TEST_CASE("PairwiseDistanceTest", "[FacilitiesTest]")
{
  arma::mat X;
  X = { { 0, 1, 1, 0, 0 },
        { 0, 1, 2, 0, 0 },
        { 1, 1, 3, 2, 0 } };
  metric::EuclideanDistance metric;
  arma::mat dist = PairwiseDistances(X, metric);
  REQUIRE(dist(0, 0) == 0);
  REQUIRE(dist(1, 0) == Approx(1.41421).epsilon(1e-5));
  REQUIRE(dist(2, 0) == 3);
}


/**
 * Test that CheckSameSizes() works in different cases.
 */
BOOST_AUTO_TEST_CASE(CheckSizeTest)
{
  arma::mat data = arma::randu<arma::mat>(20, 30);
  arma::colvec firstLabels = arma::randu<arma::colvec>(20);
  arma::colvec secondLabels = arma::randu<arma::colvec>(30);
  arma::mat thirdLabels = arma::randu<arma::mat>(20, 30);

  BOOST_REQUIRE_THROW(CheckSameSizes(data, firstLabels, "TestChecking"),
      std::invalid_argument);
  BOOST_REQUIRE_THROW(CheckSameSizes(data, firstLabels, "TestChecking", "CC"),
      std::invalid_argument);
  BOOST_REQUIRE_THROW(CheckSameSizes(data, firstLabels, "TestChecking", "AB"),
      std::runtime_error);

  BOOST_REQUIRE_NO_THROW(CheckSameSizes(data, secondLabels, "TestChecking"));
  BOOST_REQUIRE_NO_THROW(CheckSameSizes(data, thirdLabels, "TestChecking",
      "CC"));
}


/**
 * Test that CheckSameDimensionality() works in different cases.
 */
BOOST_AUTO_TEST_CASE(CheckDimensionality)
{
  arma::mat dataset = arma::randu<arma::mat>(20, 30);

  BOOST_REQUIRE_NO_THROW(CheckSameDimensionality(dataset, 20, "TestingDim"));
  BOOST_REQUIRE_NO_THROW(CheckSameDimensionality(dataset, 30, "TestingDim",
      "C"));

  BOOST_REQUIRE_THROW(CheckSameDimensionality(dataset, 100, "TestingDim"),
      std::invalid_argument);
  BOOST_REQUIRE_THROW(CheckSameDimensionality(dataset, 50, "TestingDim", "C"),
      std::invalid_argument);
  BOOST_REQUIRE_THROW(CheckSameDimensionality(dataset, 20, "TestingDim", "A"),
      std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END();
