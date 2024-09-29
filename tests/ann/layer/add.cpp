/**
 * @file tests/ann/layer/add.cpp
 * @author Ryan Curtin
 *
 * Tests the Add layer (which is just a bias).
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
 * Simple test for Add layer.
 */
TEST_CASE("AddManualWeightTestCase", "[ANNLayerTest]")
{
  arma::mat input = arma::mat(1, 1);
  input.zeros();

  arma::mat weights(1, 1);

  Add module;
  module.InputDimensions() = std::vector<size_t>({ 1 });
  module.ComputeOutputDimensions();
  module.SetWeights(weights);
  module.Parameters()[0] = 3.0;

  arma::mat output(1, 1);
  module.Forward(input, output);

  REQUIRE(output[0] == Approx(input[0] + 3.0));

  arma::mat delta(1, 1);

  // The backwards pass does not modify anything.
  module.Backward(input, output, output, delta);
  REQUIRE(delta[0] == Approx(output[0]));

  arma::mat error(1, 1);
  error(0, 0) = 2.0;
  arma::mat gradient(1, 1);
  module.Gradient(input, error, gradient);
  REQUIRE(gradient[0] == Approx(error[0]));
}

/**
 * Test the Add layer with a batch size greater than 1.
 */
TEST_CASE("AddManualWeightBatchTestCase", "[ANNLayerTest]")
{
  arma::mat input = arma::mat(1, 5);
  input.zeros();
  input[1] = 2.0;

  arma::mat weights(1, 1);

  Add module;
  module.InputDimensions() = std::vector<size_t>({ 1 });
  module.ComputeOutputDimensions();
  module.SetWeights(weights);
  module.Parameters()[0] = 3.0;

  arma::mat output(1, 5);
  module.Forward(input, output);

  REQUIRE(output[0] == Approx(input[0] + 3.0));
  REQUIRE(output[1] == Approx(input[1] + 3.0));
  REQUIRE(output[2] == Approx(input[2] + 3.0));
  REQUIRE(output[3] == Approx(input[3] + 3.0));
  REQUIRE(output[4] == Approx(input[4] + 3.0));

  arma::mat delta(1, 5);

  // The backwards pass does not modify anything.
  module.Backward(input, output, output, delta);
  REQUIRE(delta[0] == Approx(output[0]));
  REQUIRE(delta[1] == Approx(output[1]));
  REQUIRE(delta[2] == Approx(output[2]));
  REQUIRE(delta[3] == Approx(output[3]));
  REQUIRE(delta[4] == Approx(output[4]));

  arma::mat error(1, 5);
  error.fill(2.0);
  error(0, 1) = 3.0;
  arma::mat gradient(1, 1);
  module.Gradient(input, error, gradient);
  REQUIRE(gradient[0] == Approx(11.0));
}
