/**
 * @file tests/ann/layer/softshrink.cpp
 * @author Aditya Raj
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
 * Simple SoftShrink module test.
 */
TEST_CASE("SimpleSoftShrinkLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output, gy, g;
  arma::mat calculatedOutput;
  const double lambda = 0.7;
  SoftShrink module(lambda);

  // Test the forward function.
  input = {{0.92, 0.67, -0.99, 0.52},
            {0.83, -0.41, 0.71, -1.29},
            {0.88, 0.44, 0.14, -3.67},
            {0.22, -0.56, 0.92, -0.35}};

  calculatedOutput = {{0.22, 0, -0.29, 0},
                      {0.13, 0, 0.01, -0.59},
                      {0.18, 0, 0, -2.97},
                      {0, 0, 0.22, 0}};
  module.Forward(input, output);

  REQUIRE(arma::accu(arma::abs(calculatedOutput - output)) ==
      Approx(0.0).margin(1e-4));

  // Test the backward function.
  gy = arma::zeros(input.n_rows, input.n_cols);
  gy(0) = 1;
  gy(3) = 1;
  gy(6) = 1;
  gy(8) = 1;
  gy(11) = 1;

  arma::mat calculatedGradient = {{1.0, 0, 1.0, 0},
                                  {0, 0, 0, 0},
                                  {0, 0, 0, 0},
                                  {0, 0, 1.0, 0}};
  module.Backward(output, gy, g);

  REQUIRE(arma::accu(arma::abs(calculatedGradient - g)) ==
      Approx(0.0).margin(1e-04));
}