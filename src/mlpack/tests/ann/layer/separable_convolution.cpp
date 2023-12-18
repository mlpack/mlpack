/**
 * @file tests/ann/layer/separable_convolution.cpp
 * @author Aakash Kaushik
 * @author Sidharth
 *
 * Tests the separable convolution layer modules.
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
 * Simple Separable Convolution layer test.
 */
TEST_CASE("SeparableConvolutionShapeAndSizeTest", "[ANNLayerTest]") {

  size_t inSize = 3;
  size_t outSize = 16;
  size_t kernelWidth = 3;
  size_t kernelHeight = 3;
  size_t strideWidth = 1;
  size_t strideHeight = 1;
  size_t padW = 1;
  size_t padH = 1;
  size_t inputWidth = 32;
  size_t inputHeight = 30;
  size_t numGroups = 1; 

  SeparableConvolution module(inSize,
                               outSize,
                               kernelWidth,
                               kernelHeight,
                               strideWidth,
                               strideHeight,
                               padW,
                               padH,
                               inputWidth,
                               inputHeight,
                               numGroups);

  /**
   * generating a random input
  */
  arma::mat input;
  input.set_size(inputWidth * inputHeight * inSize, 1);

  arma::mat output;
  module.Forward(input, output);

  /**
   * comparing the input and output size
  */
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  
  /**
   * generating random gradients for the backward pass
  */
  arma::mat gradient;
  gradient.set_size(output.n_rows, output.n_cols);
  gradient.randn();
  arma::mat delta;
  module.Backward(input, output, gradient, delta);

  REQUIRE(delta.n_rows == input.n_rows);
  REQUIRE(delta.n_cols == input.n_cols);
  
  /**
   * generating random error values for gradient test
  */
  arma::mat error;
  error.set_size(output.n_rows, output.n_cols);
  error.randn();
  arma::mat computedGradient;
  module.Gradient(input, error, computedGradient);

  // Test for expected gradient size and shape.
  REQUIRE(computedGradient.n_rows == module.Parameters().n_rows);
  REQUIRE(computedGradient.n_cols == module.Parameters().n_cols);
}
