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
TEST_CASE("SeparableConvolutionShapeAndSizeTest", "[ANNLayerTest]") 
{
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
   * comparing the input and output size for forward test
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

/**
 * Jacobian test for the Separable Convolution layer
*/
TEST_CASE("CustomJacobianSeparableConvolutionTest", "[ANNLayerTest]")
{
  const size_t inSize = 3;
  const size_t outSize = 5;
  const size_t kernelWidth = 3;
  const size_t kernelHeight = 3;
  const size_t strideWidth = 1;
  const size_t strideHeight = 1;
  const size_t padWLeft = 1;
  const size_t padWRight = 1;
  const size_t padHTop = 1;
  const size_t padHBottom = 1;
  const size_t inputWidth = 5;
  const size_t inputHeight = 5;
  const size_t numGroups = 1;

  // Initialize the separable convolution layer.
  SeparableConvolution layer(inSize,
                             outSize,
                             kernelWidth, 
                             kernelHeight,
                             strideWidth, 
                             strideHeight, 
                             padWLeft, 
                             padWRight,
                             padHTop, 
                             padHBottom, 
                             inputWidth, 
                             inputHeight,
                             numGroups);
  /**
   * Random input for Jacobian test.
   */
  arma::mat input = arma::randu(3, 5);
  double error = CustomJacobianTest(layer, input);
  /**
   * Check whether the error is below a certain threshold.
   */
  REQUIRE(error <= 1e-5);
}

/**
 * SeparableConvolution layer numerical gradient test.
 */
TEST_CASE("GradientSeparableConvolutionTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(32, 64)), 
        target(arma::zeros(10, 64)) 
    {
      const size_t inSize = 3;
      const size_t outSize = 5;
      const size_t kernelWidth = 3;
      const size_t kernelHeight = 3;
      const size_t strideWidth = 1;
      const size_t strideHeight = 1;
      const size_t padWLeft = 1;
      const size_t padWRight = 1;
      const size_t padHTop = 1;
      const size_t padHBottom = 1;
      const size_t inputWidth = 5;
      const size_t inputHeight = 5;
      const size_t numGroups = 1;

  // Initialize the separable convolution layer.
      SeparableConvolution layer(inSize,
                                outSize,
                                kernelWidth, 
                                kernelHeight,
                                strideWidth, 
                                strideHeight, 
                                padWLeft, 
                                padWRight,
                                padHTop, 
                                padHBottom, 
                                inputWidth, 
                                inputHeight,
                                numGroups);

      model = new FFN<NegativeLogLikelihood, XavierInitialization>();
      model->ResetData(input, target);
      model->Add<SeparableConvolution>(layer);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 64);
      model->Gradient(model->Parameters(), 0, gradient, 64);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, XavierInitialization>* model;
    arma::mat input, target;
  } function;

  double gradient = CheckGradient(function);

  REQUIRE(gradient < 1e-1);
}
