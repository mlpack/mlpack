/*
 * @file tests/ann/layer/ftswish.cpp
 * @author Mayank Raj
 *
 * Tests the FTSwish layer.
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
 * Simple test case for the FTSwish layer.
 */
TEST_CASE("FTSwishTest", "[ANNLayerTest]")
{
  // Set the threshold value for the FTSwish layer.
  double threshold = -0.2;

  // Create the FTSwish layer.
  FTSwishType<> layer(threshold);

  // Input and output matrices.
  arma::mat input = {{0.234, 1.23, -1.34},
    {1.45, 2.001, -0.98},
    {-3.14, 3.43, 9.9}};
  arma::mat actualOutput = {{-0.06937312,  0.75179685, -0.2},
    {0.97449773,  1.56268497, -0.2},
    {-0.2,  3.1223977, 9.6995033 }};
  arma::mat output;
  output.set_size(3, 3);
  // Forward pass.
  layer.Forward(input, output);

  // Test the Forward function
  REQUIRE(abs(accu(output - actualOutput)) <= 0.0001);

  arma::mat delta = {{0.615941, 0.989097, 0.0},
    {1.03315, 1.09083, 0.0},
    {0.0, 1.07286, 1.00045}};

  arma::mat gy, g;
  gy.set_size(3, 3);
  gy.fill(1);
  g.set_size(3, 3);
  // Backward pass.
  layer.Backward(input, output, gy, g);

  // Test the Backward function.
  REQUIRE(abs(accu(g - delta)) <= 0.0001);
}
