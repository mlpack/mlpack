/**
 * @file performance_functions_test.cpp
 * @author Marcus Edel
 *
 *  Tests for the various performance functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>
#include <mlpack/methods/ann/performance_functions/sse_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(PerformanceFunctionsTest);

// Test the mean squared error performance function.
BOOST_AUTO_TEST_CASE(MeanSquaredErrorTest)
{
  arma::colvec input("1.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0");
  arma::colvec target = arma::zeros<arma::colvec>(8);

  BOOST_REQUIRE_EQUAL(MeanSquaredErrorFunction::Error(input, target), 0.5);
}

// Test the cross entropy performance function.
BOOST_AUTO_TEST_CASE(CrossEntropyErrorTest)
{
  arma::colvec input;
  input << std::exp(-2.0) << std::exp(-1.0);
  arma::colvec target = arma::ones<arma::colvec>(2);

  BOOST_REQUIRE_EQUAL(CrossEntropyErrorFunction<>::Error(input, target), 3);
}

// Test the sum squared error performance function.
BOOST_AUTO_TEST_CASE(SumSquaredErrorTest)
{
  arma::colvec input("1.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0");
  arma::colvec target = arma::zeros<arma::colvec>(8);

  BOOST_REQUIRE_EQUAL(SumSquaredErrorFunction::Error(input, target), 4);
}

BOOST_AUTO_TEST_SUITE_END();
