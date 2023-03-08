/**
 * @file tests/ann/layer/join.cpp
 * @author Marcus Edel
 *
 * Tests the ann layer modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

/**
 * Simple join module test.
 */
TEST_CASE("SimpleJoinLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  input = arma::ones(10, 5);

  // Test the Forward function.
  Join module;
  module.Forward(input, output);
  REQUIRE(50 == arma::accu(output));

  bool b = output.n_rows == 1 || output.n_cols == 1;
  REQUIRE(b == true);

  // Test the Backward function.
  module.Backward(input, output, delta);
  REQUIRE(50 == arma::accu(delta));

  b = delta.n_rows == input.n_rows && input.n_cols;
  REQUIRE(b == true);
}
