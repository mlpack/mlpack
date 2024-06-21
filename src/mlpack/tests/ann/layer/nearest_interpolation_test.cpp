/**
 * @file tests/ann/layer/add.cpp
 * @author Ryan Curtin
 *
 * Tests the nearest interpolation layer
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
 * Simple test for the NearestInterpolation layer
 */
TEST_CASE("SimpleNearestInterpolationLayerTest", "[ANNLayerTest]")
{
  // Tested output against torch.nn.Upsample(mode="nearest").
  arma::mat input, output, unzoomedOutput, expectedOutput;
  size_t inRowSize = 2;
  size_t inColSize = 2;
  size_t outRowSize = 5;
  size_t outColSize = 7;
  size_t depth = 1;
  input.zeros(inRowSize * inColSize * depth, 1);
  input[0] = 1.0;
  input[1] = 3.0;
  input[2] = 2.0;
  input[3] = 4.0;
  NearestInterpolation layer(inRowSize, inColSize, outRowSize,
                               outColSize, depth);

  expectedOutput << 1.0000 << 1.0000 << 1.0000 << 1.0000 << 2.0000
                 << 2.0000 << 2.0000 << arma::endr
                 << 1.0000 << 1.0000 << 1.0000 << 1.0000 << 2.0000
                 << 2.0000 << 2.0000 << arma::endr
                 << 1.0000 << 1.0000 << 1.0000 << 1.0000 << 2.0000
                 << 2.0000 << 2.0000 << arma::endr
                 << 3.0000 << 3.0000 << 3.0000 << 3.0000 << 4.0000
                 << 4.0000 << 4.0000 << arma::endr
                 << 3.0000 << 3.0000 << 3.0000 << 3.0000 << 4.0000
                 << 4.0000 << 4.0000 << arma::endr;
  expectedOutput.reshape(35, 1);

  layer.Forward(input, output);
  CheckMatrices(output - expectedOutput,
                arma::zeros(output.n_rows), 1e-4);

  expectedOutput.clear();
  expectedOutput << 12.0000 << 18.0000 << arma::endr
                 << 24.0000 << 24.0000 << arma::endr;
  expectedOutput.reshape(4, 1);
  layer.Backward(output, output, unzoomedOutput);
  CheckMatrices(unzoomedOutput - expectedOutput,
      arma::zeros(input.n_rows), 1e-4);

  arma::mat input1, output1, unzoomedOutput1, expectedOutput1;
  inRowSize = 2;
  inColSize = 3;
  outRowSize = 17;
  outColSize = 23;
  input1 << 1 << 2 << 3 << arma::endr
         << 4 << 5 << 6 << arma::endr;
  input1.reshape(6, 1);
  NearestInterpolation layer1(inRowSize, inColSize, outRowSize,
                                outColSize, depth);

  layer1.Forward(input1, output1);
  layer1.Backward(output1, output1, unzoomedOutput1);

  REQUIRE(accu(output1) - 1317.00 == Approx(0.0).margin(1e-05));
  REQUIRE(accu(unzoomedOutput1) - 1317.00 ==
          Approx(0.0).margin(1e-05));
}
