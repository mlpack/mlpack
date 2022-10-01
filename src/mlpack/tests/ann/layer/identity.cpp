/**
 * @file tests/ann_layer_test.cpp
 * @author Marcus Edel
 * @author Praveen Ch
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
 * Simple test for Identity layer.
 */
TEST_CASE("IdentityTestCase", "[ANNLayerTest]")
{
  // For rectangular input to pooling layers.
  arma::mat input = arma::mat(12, 1, arma::fill::randn);
  arma::mat output;
  // Output-Size should be 4 x 3.
  output.set_size(12, 1);

  Identity module1;
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  module1.Forward(input, output);
  CheckMatrices(output, input, 1e-1);
  REQUIRE(output.n_elem == 12);
  REQUIRE(output.n_cols == 1);
  REQUIRE(input.memptr() != output.memptr());

  arma::mat prevDelta = arma::mat(12, 1, arma::fill::randn);
  arma::mat delta;
  delta.set_size(12, 1);
  module1.Backward(input, prevDelta, delta);
  CheckMatrices(delta, prevDelta, 1e-1);
  REQUIRE(delta.memptr() != prevDelta.memptr());
}
