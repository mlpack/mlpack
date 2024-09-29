/**
 * @file tests/digamma_test.cpp
 * @author Gopi Tatiraju
 * 
 * Test the digamma distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;

/**
 * Test the output of digamma for large input values.
 */
TEST_CASE("DigammaLarge", "[DigammaTest]")
{
  arma::mat data;

  if (!data::Load("digamma_data.csv", data, true, false))
    FAIL("Cannot load data digamma_data.csv");
  
  for (size_t i = 0; i < data.n_rows; i++)
    REQUIRE(Digamma(data(i, 0)) == Approx(data(i, 1)).epsilon(1e-7));
}

/**
 * Test the output of digamma for negative input values.
 */
TEST_CASE("DigammaNegative", "[DigammaTest]")
{
  arma::mat data;

  if (!data::Load("digamma_neg_data.csv", data, true, false))
    FAIL("Cannot load data digamma_neg_data.csv");
  
  for (size_t i = 0; i < data.n_rows; i++)
    REQUIRE(Digamma(data(i, 0)) == Approx(data(i, 1)).epsilon(1e-7));
}

/**
 * Test the output of digamma for small input values.
 */
TEST_CASE("DigammaSmall", "[DigammaTest]")
{
  arma::mat data;

  if (!data::Load("digamma_small_data.csv", data, true, false))
    FAIL("Cannot load data digamma_small_data.csv");
  
  for (size_t i = 0; i < data.n_rows; i++)
    REQUIRE(Digamma(data(i, 0)) == Approx(data(i, 1)).epsilon(1e-7));
}

/**
 * Test the output of digamma for values near the positive roots.
 */
TEST_CASE("DigammaNearPositiveRoots", "[DigammaTest]")
{
  arma::mat data;

  if (!data::Load("digamma_root_data.csv", data, true, false))
    FAIL("Cannot load data digamma_root_data.csv");
  
  for (size_t i = 0; i < data.n_rows; i++)
    REQUIRE(Digamma(data(i, 0)) == Approx(data(i, 1)).epsilon(1e-7));
}
