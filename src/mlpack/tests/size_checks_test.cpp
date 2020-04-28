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
#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::util;
BOOST_AUTO_TEST_SUITE(SizeCheckTest);

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
  arma::colvec refSet = arma::randu<arma::colvec>(20);
  arma::colvec refSet2 = arma::randu<arma::colvec>(40);

  BOOST_REQUIRE_NO_THROW(CheckSameDimensionality(dataset, (size_t) 20,
      "TestingDim"));
  BOOST_REQUIRE_THROW(CheckSameDimensionality(dataset, (size_t) 100,
      "TestingDim"), std::invalid_argument);

  BOOST_REQUIRE_THROW(CheckSameDimensionality(dataset, refSet2, "TestingDim"),
      std::invalid_argument);
  BOOST_REQUIRE_NO_THROW(CheckSameDimensionality(dataset, refSet,
      "TestingDim"));
}

BOOST_AUTO_TEST_SUITE_END();

