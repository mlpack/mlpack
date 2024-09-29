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
TEST_CASE("NearestInterpolationLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output, unzoomedOutput, expectedOutput;

  size_t inColSize = 2;
  size_t inRowSize = 2;
  size_t channels = 1;

  double scaleFactor = 2.0f;

  input.zeros(inRowSize * inColSize, channels);
  output.zeros(inRowSize * scaleFactor * inColSize * scaleFactor, channels);
  unzoomedOutput = input;
  input[0] = 1.0;
  input[1] = 2.0;
  input[2] = 3.0;
  input[3] = 4.0;

  mlpack::NearestInterpolation layer;


  layer = mlpack::NearestInterpolation({scaleFactor, scaleFactor});

  layer.InputDimensions() = { inRowSize, inColSize, channels };
  layer.ComputeOutputDimensions();
  
  expectedOutput  << 1.0000 << 1.0000 << 2.0000 << 2.0000
		  << 1.0000 << 1.0000 << 2.0000 << 2.0000
		  << 3.0000 << 3.0000 << 4.0000 << 4.0000
		  << 3.0000 << 3.0000 << 4.0000 << 4.0000 << arma::endr;

  expectedOutput.reshape(16, 1);
  layer.Forward(input, output);
  CheckMatrices(output - expectedOutput,
                arma::zeros(output.n_rows), 1e-4);

  expectedOutput.clear();
  expectedOutput << 4.0000 << 8.0000
                 << 12.0000 << 16.0000 << arma::endr;
  expectedOutput.reshape(4, 1);
  layer.Backward(output, output, unzoomedOutput);
  CheckMatrices(unzoomedOutput - expectedOutput,
      arma::zeros(input.n_rows), 1e-4);

  arma::mat input1, output1, unzoomedOutput1, expectedOutput1;
  inRowSize = 2;
  inColSize = 3;

  input1 << 1 << 2 << 3 << arma::endr
         << 4 << 5 << 6 << arma::endr;
  input1.reshape(6, 1);
  output1.zeros(17*23, 1);
  unzoomedOutput1.zeros(6, 1);
  mlpack::NearestInterpolation layer1({17/2.0f, 23/3.0f});

  layer1.InputDimensions() = { 2, 3, channels };
  layer1.ComputeOutputDimensions();

  layer1.Forward(input1, output1);
  layer1.Backward(output1, output1, unzoomedOutput1);
  REQUIRE(accu(output1) - 1317.00 == Approx(0.0).margin(1e-05));
  REQUIRE(accu(unzoomedOutput1) - 1317.00 ==
          Approx(0.0).margin(1e-05));
}
