/**
 * @file size_checks_test.cpp
 * @author Bisakh Mondal
 *
 * Test file for Utility size_checks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::util;

/**
 * Test that CheckSameSizes() works in different cases.
 */
TEST_CASE("CheckSizeTest", "[SizeCheckTest]")
{
  arma::mat data = arma::randu<arma::mat>(20, 30);
  arma::rowvec firstLabels = arma::randu<arma::rowvec>(20);
  arma::rowvec secondLabels = arma::randu<arma::rowvec>(30);
  arma::mat thirdLabels = arma::randu<arma::mat>(40, 30);

  REQUIRE_THROWS_AS(CheckSameSizes(data, firstLabels, "TestChecking"),
      std::invalid_argument);
  REQUIRE_THROWS_AS(CheckSameSizes(data, (size_t) 20, "TestChecking"),
      std::invalid_argument);

  REQUIRE_NOTHROW(CheckSameSizes(data, secondLabels, "TestChecking"));
  REQUIRE_NOTHROW(CheckSameSizes(data, (size_t) 30, "TestChecking"));
  REQUIRE_NOTHROW(CheckSameSizes(data, (size_t) thirdLabels.n_cols,
      "TestChecking"));
}

/**
 * Test that CheckSameDimensionality() works in different cases.
 */
TEST_CASE("CheckDimensionality", "[SizeCheckTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(20, 30);
  arma::colvec refSet = arma::randu<arma::colvec>(20);
  arma::colvec refSet2 = arma::randu<arma::colvec>(40);

  REQUIRE_NOTHROW(CheckSameDimensionality(dataset, (size_t) 20,
      "TestingDim"));
  REQUIRE_THROWS_AS(CheckSameDimensionality(dataset, (size_t) 100,
      "TestingDim"), std::invalid_argument);

  REQUIRE_THROWS_AS(CheckSameDimensionality(dataset, refSet2, "TestingDim"),
      std::invalid_argument);
  REQUIRE_NOTHROW(CheckSameDimensionality(dataset, refSet,
      "TestingDim"));
}
