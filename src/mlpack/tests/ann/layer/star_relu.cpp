/*
 * @file tests/ann/layer/star_relu.cpp
 * @author Mayank Raj
 *
 * Tests the StarReLU layer.
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
 * Simple test case for the StaReLU layer.
 */
TEST_CASE("StarReLUTest", "[ANNLayerTest]")
{
  // Set the parameters for the StarReLU layer.
  double s = 2.0;
  double b = 1.0;

  // Create the StarReLU layer.
  StarReLUType<> layer(s, b);

  // Input and output matrices.
  arma::mat input = {{-0.5, 1.0, 0.2},
                     {2.0, -0.3, -1.0},
                     {-1.5, 0.4, 0.7}};
  arma::mat actualOutput = {{1.0, 3.0, 1.08},
                            {9.0, 1.0, 1.0},
                            {1.0, 1.32, 1.98}};
  arma::mat output;
  output.set_size(3, 3);

  // Forward pass.
  layer.Forward(input, output);

  // Test the Forward function.
  REQUIRE(arma::approx_equal(output, actualOutput, "absdiff", 1e-5));

  arma::mat delta = {{4, 12, 4.32},
                     {36, 4, 4},
                     {4, 5.28, 7.92}};
  arma::mat gy, g;
  gy.set_size(3, 3);
  gy.fill(1.0);
  g.set_size(3, 3);

  // Backward pass.
  layer.Backward(output, gy, g);

  // Test the Backward function.
  REQUIRE(arma::approx_equal(g, delta, "absdiff", 1e-5));
}

/**
 * JacobianTest for StarRelu layer
 */
TEST_CASE("JacobianStarReluLayerTest", "[ANNLayerTest]")
{
    const size_t elems = arma::randi(arma::distr_param(2, 1000));

    arma::mat input(elems, 1);

    StarReLU module;
    module.InputDimensions() = { elems };
    module.ComputeOutputDimensions();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
}