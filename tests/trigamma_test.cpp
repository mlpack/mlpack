/**
 * @file tests/digamma_test.cpp
 * @author Gopi Tatiraju
 * 
 * Test the trigamma function.
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
 * Test the output of trigamma for input values.
 */
TEST_CASE("Trigamma", "[TrigammaTest]")
{
  arma::mat data;

  if (!data::Load("trigamma_data.csv", data, true, false))
    FAIL("Cannot load data trigamma_data.csv");

  for (size_t i = 0; i < data.n_rows; i++)
    REQUIRE(Trigamma(data(i, 0)) == Approx(data(i, 1)).epsilon(1e-7));
}

