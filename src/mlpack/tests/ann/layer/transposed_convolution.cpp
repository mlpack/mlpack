
/**
 * @file tests/ann/layer/transposed_convolution.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 * @author Ranjodh Singh
 *
 * Test the TransposedConvolution layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/transposed_convolution.hpp>

#include "../../catch.hpp"
#include "../../test_catch_tools.hpp"
#include "../ann_test_tools.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

/**
 * Test that the functions that can modify and access the parameters of the
 * Transposed Convolution layer work.
 */
TEST_CASE("TransposedConvolutionParametersTest", "[ANNLayerTest]")
{
  // Parameter order: outSize, kW, kH, dW, dH, padW, padH, paddingType.
  TransposedConvolution layer1(2, 3, 4, 5, 6, std::tuple<size_t, size_t>(7, 8),
      std::tuple<size_t, size_t>(9, 10), "none");
  TransposedConvolution layer2(3, 4, 5, 6, 7, std::tuple<size_t, size_t>(8, 9),
      std::tuple<size_t, size_t>(10, 11), "none");

  // Make sure we can get the parameters successfully.
  REQUIRE(layer1.KernelWidth() == 3);
  REQUIRE(layer1.KernelHeight() == 4);
  REQUIRE(layer1.StrideWidth() == 5);
  REQUIRE(layer1.StrideHeight() == 6);
  REQUIRE(layer1.PadWLeft() == 7);
  REQUIRE(layer1.PadWRight() == 8);
  REQUIRE(layer1.PadHTop() == 9);
  REQUIRE(layer1.PadHBottom() == 10);

  // Now modify the parameters to match the second layer.
  layer1.KernelWidth() = 4;
  layer1.KernelHeight() = 5;
  layer1.StrideWidth() = 6;
  layer1.StrideHeight() = 7;
  layer1.PadWLeft() = 8;
  layer1.PadWRight() = 9;
  layer1.PadHTop() = 10;
  layer1.PadHBottom() = 11;

  // Now ensure all results are the same.
  REQUIRE(layer1.KernelWidth() == layer2.KernelWidth());
  REQUIRE(layer1.KernelHeight() == layer2.KernelHeight());
  REQUIRE(layer1.StrideWidth() == layer2.StrideWidth());
  REQUIRE(layer1.StrideHeight() == layer2.StrideHeight());
  REQUIRE(layer1.PadWLeft() == layer2.PadWLeft());
  REQUIRE(layer1.PadWRight() == layer2.PadWRight());
  REQUIRE(layer1.PadHTop() == layer2.PadHTop());
  REQUIRE(layer1.PadHBottom() == layer2.PadHBottom());
}

/**
 * Test the functions that compute dimensions for
 * Transposed Convolution layer.
 */
TEST_CASE("TransposedConvolutionDimensionsTest", "[ANNLayerTest]")
{
  struct Config
  {
    std::string paddingType;
    size_t inputWidth;
    size_t inputHeight;
    size_t kernelWidth;
    size_t kernelHeight;
    size_t strideWidth;
    size_t strideHeight;
    size_t padWLeft;
    size_t padWRight;
    size_t padHTop;
    size_t padHBottom;
    size_t expecedOutputWidth;
    size_t expecedOutputHeight;
  };

  const std::vector<Config> configs = {
    // VALID (pads always treated as zero)
    { "valid",  7,  7, 3, 3, 1, 1, 0, 0, 0, 0,  9,  9 },
    { "valid",  7,  8, 4, 2, 1, 2, 0, 0, 0, 0, 10, 16 },
    { "valid",  8,  7, 5, 1, 2, 1, 0, 0, 0, 0, 19,  7 },
    { "valid",  8,  8, 2, 5, 3, 3, 0, 0, 0, 0, 23, 26 },
    { "valid",  6,  9, 4, 4, 2, 1, 0, 0, 0, 0, 14, 12 },
    { "valid",  9,  6, 3, 5, 1, 3, 0, 0, 0, 0, 11, 20 },
    { "valid", 10, 10, 5, 2, 3, 1, 0, 0, 0, 0, 32, 11 },
    { "valid",  5,  5, 2, 3, 1, 2, 0, 0, 0, 0,  6, 11 },
    { "valid",  5,  6, 3, 4, 2, 2, 0, 0, 0, 0, 11, 14 },

    // SAME (output spatial == input spatial)
    { "same",  7,  7, 3, 3, 1, 1, 0, 0, 0, 0,  7,  7 },
    { "same",  7,  8, 4, 2, 2, 2, 0, 0, 0, 0,  7,  8 },
    { "same",  8,  7, 5, 1, 1, 3, 0, 0, 0, 0,  8,  7 },
    { "same",  8,  8, 2, 5, 3, 1, 0, 0, 0, 0,  8,  8 },
    { "same",  6,  9, 4, 4, 2, 3, 0, 0, 0, 0,  6,  9 },
    { "same",  9,  6, 3, 5, 1, 2, 0, 0, 0, 0,  9,  6 },
    { "same", 10, 10, 5, 2, 1, 4, 0, 0, 0, 0, 10, 10 },
    { "same",  5,  5, 2, 3, 2, 1, 0, 0, 0, 0,  5,  5 },

    // NONE (explicit pads)
    { "none",  7,  7, 3, 3, 1, 1, 1, 1, 1, 1,  7,  7 },
    { "none",  8,  8, 4, 2, 2, 2, 1, 2, 2, 1, 15, 13 },
    { "none",  7,  8, 5, 3, 1, 2, 0, 1, 1, 0, 10, 16 },
    { "none",  8,  7, 2, 5, 3, 1, 2, 0, 0, 2, 21,  9 },
    { "none",  6,  6, 3, 4, 2, 3, 1, 1, 2, 0, 11, 17 },
    { "none",  9,  5, 4, 2, 1, 3, 0, 3, 1, 1,  9, 12 },
    { "none",  5,  9, 5, 5, 2, 2, 2, 2, 2, 2,  9, 17 },
    { "none", 10, 10, 3, 3, 3, 3, 1, 4, 0, 3, 25, 27 }
  };

  const size_t inMaps = 1, maps = 2;
  for (size_t i = 0; i < configs.size(); ++i)
  {
    const auto& c = configs[i];
    std::ostringstream sectionName;
    sectionName << "Config " << i << ": paddingType=" << c.paddingType
                << ", input=" << c.inputWidth << "x" << c.inputHeight
                << ", kernel=" << c.kernelWidth << "x" << c.kernelHeight
                << ", stride=" << c.strideWidth << "x" << c.strideHeight
                << ", padW=" << c.padWLeft << "," << c.padWRight
                << ", padH=" << c.padHTop << "," << c.padHBottom;
    SECTION(sectionName.str())
    {
      TransposedConvolution module(
          maps,
          c.kernelWidth,
          c.kernelHeight,
          c.strideWidth,
          c.strideHeight,
          std::make_tuple(c.padWLeft, c.padWRight),
          std::make_tuple(c.padHTop, c.padHBottom),
          c.paddingType);

      module.InputDimensions() = { c.inputWidth, c.inputHeight, inMaps };
      module.ComputeOutputDimensions();

      // WeightSize = inMaps * maps * kernelWidth * kernelHeight + maps (bias).
      size_t expectedWeightSize = inMaps * maps * c.kernelWidth
          * c.kernelHeight + maps;
      REQUIRE(module.WeightSize() == expectedWeightSize);

      // OutputSize = (outputWidth * outputHeight * maps).
      size_t expectedOutputSize = c.expecedOutputWidth * c.expecedOutputHeight
          * maps;
      REQUIRE(module.OutputSize() == expectedOutputSize);
    }
  }
}

/**
 * Test The Functions That Set Weights of Transposed Convolution Layer.
 */
TEST_CASE("TransposedConvolutionWeightInitializationTest", "[ANNLayerTest]")
{
  size_t maps = 3;
  size_t inMaps = 3;
  size_t kernelWidth = 4, kernelHeight = 4;
  TransposedConvolution module = TransposedConvolution(
      maps,
      kernelWidth,
      kernelHeight,
      1,
      1,
      1,
      1);
  module.InputDimensions() = std::vector<size_t>({5, 5, inMaps});
  module.ComputeOutputDimensions();

  arma::mat weights = arma::zeros(kernelWidth*kernelHeight*inMaps*maps+maps);
  module.SetWeights(weights);

  REQUIRE(std::equal(module.Weight().begin(),
      module.Weight().end(), module.Parameters().begin()));
  REQUIRE(std::equal(module.Bias().begin(),
      module.Bias().end(), module.Parameters().end() - maps));
  REQUIRE(module.Weight().n_rows == kernelWidth);
  REQUIRE(module.Weight().n_cols == kernelHeight);
  REQUIRE(module.Weight().n_slices == inMaps * maps);
  REQUIRE(module.Bias().n_rows == maps);
  REQUIRE(module.Bias().n_cols == 1);
  REQUIRE(module.Parameters().n_rows ==
      (maps * inMaps * kernelWidth * kernelHeight) + maps);
}

/**
 * Test the Forward and Backward pass of the Transposed Convolution layer.
 **/
TEST_CASE("TransposedConvolutionForwardBackwardTest", "[ANNLayerTest]")
{
  arma::mat input, output, weights, delta;

  SECTION("3x3 Kernel, stride=1, no padding")
  {
    TransposedConvolution module(1, 3, 3, 1, 1, 0, 0);
    input = arma::linspace<arma::colvec>(0, 15, 16);
    module.InputDimensions() = {4, 4};
    module.ComputeOutputDimensions();

    weights.set_size(module.WeightSize(), 1);
    weights.zeros();
    weights[0] = 1.0;
    weights[8] = 2.0;
    module.SetWeights(weights);

    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);

    delta.set_size(arma::size(input));
    module.Backward(input, output, output, delta);

    REQUIRE(accu(output) == 360.0);
    REQUIRE(accu(delta) == 720.0);
    REQUIRE(weights.n_elem == 9 + 1);
  }

  SECTION("4x4 kernel, pad=1, stride=1")
  {
    TransposedConvolution module(1, 4, 4, 1, 1, 1, 1);
    input = arma::linspace<arma::colvec>(0, 24, 25);
    module.InputDimensions() = {5, 5};
    module.ComputeOutputDimensions();

    weights.set_size(module.WeightSize(), 1);
    weights.zeros();
    weights[0] = 1.0;
    weights[3] = 1.0;
    weights[6] = 1.0;
    weights[9] = 1.0;
    weights[12] = 1.0;
    weights[15] = 2.0;
    module.SetWeights(weights);

    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);

    delta.set_size(arma::size(input));
    module.Backward(input, output, output, delta);

    REQUIRE(accu(delta) == 6504.0);
    REQUIRE(accu(output) == 1512.0);
    REQUIRE(weights.n_elem == 16 + 1);
  }

  SECTION("3x3 kernel, pad=1, stride=1")
  {
    TransposedConvolution module(1, 3, 3, 1, 1, 1, 1);
    input = arma::linspace<arma::colvec>(0, 24, 25);
    module.InputDimensions() = {5, 5};
    module.ComputeOutputDimensions();

    weights.set_size(module.WeightSize(), 1);
    weights.zeros();
    weights[1] = 2.0;
    weights[2] = 4.0;
    weights[3] = 3.0;
    weights[8] = 1.0;
    module.SetWeights(weights);

    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);

    delta.set_size(arma::size(input));
    module.Backward(input, output, output, delta);

    REQUIRE(accu(delta) == 19154.0);
    REQUIRE(accu(output) == 2370.0);
    REQUIRE(weights.n_elem == 9 + 1);
  }

  SECTION("3x3 kernel, stride=1, no padding")
  {
    TransposedConvolution module(1, 3, 3, 1, 1, 0, 0);
    input = arma::linspace<arma::colvec>(0, 24, 25);
    module.InputDimensions() = {5, 5};
    module.ComputeOutputDimensions();

    weights.set_size(module.WeightSize(), 1);
    weights.zeros();
    weights[2] = 2.0;
    weights[4] = 4.0;
    weights[6] = 6.0;
    weights[8] = 8.0;
    module.SetWeights(weights);

    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);

    delta.set_size(arma::size(input));
    module.Backward(input, output, output, delta);

    REQUIRE(accu(delta) == 86208.0);
    REQUIRE(accu(output) == 6000.0);
    REQUIRE(weights.n_elem == 9 + 1);
  }

  SECTION("3x3 kernel, stride=2, no padding")
  {
    TransposedConvolution module(1, 3, 3, 2, 2, 0, 0);
    input = arma::linspace<arma::colvec>(0, 3, 4);
    module.InputDimensions() = {2, 2};
    module.ComputeOutputDimensions();

    weights.set_size(module.WeightSize(), 1);
    weights.zeros();
    weights[2] = 8.0;
    weights[4] = 6.0;
    weights[6] = 4.0;
    weights[8] = 2.0;
    module.SetWeights(weights);

    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);

    delta.set_size(arma::size(input));
    module.Backward(input, output, output, delta);

    REQUIRE(accu(delta) == 960.0);
    REQUIRE(accu(output) == 120.0);
    REQUIRE(weights.n_elem == 9 + 1);
  }

  SECTION("3x3 kernel, stride=2, pad=1")
  {
    TransposedConvolution module(1, 3, 3, 2, 2, 1, 1);
    input = arma::linspace<arma::colvec>(0, 8, 9);
    module.InputDimensions() = {3, 3};
    module.ComputeOutputDimensions();

    weights.set_size(module.WeightSize(), 1);
    weights.zeros();

    weights[0] = 8.0;
    weights[3] = 6.0;
    weights[6] = 2.0;
    weights[8] = 4.0;
    module.SetWeights(weights);

    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);

    delta.set_size(arma::size(input));
    module.Backward(input, output, output, delta);

    REQUIRE(accu(output) == 410.0);
    REQUIRE(accu(delta) == 4444.0);
    REQUIRE(weights.n_elem == 9 + 1);
  }

  SECTION("3x3 kernel, stride=2, pad=1")
  {
    TransposedConvolution module(1, 3, 3, 2, 2, 1, 1);
    input = arma::linspace<arma::colvec>(0, 8, 9);
    module.InputDimensions() = {3, 3};
    module.ComputeOutputDimensions();

    weights.set_size(module.WeightSize(), 1);
    weights.zeros();
    weights[0] = 8.0;
    weights[2] = 6.0;
    weights[4] = 2.0;
    weights[8] = 4.0;
    module.SetWeights(weights);

    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);

    delta.set_size(arma::size(input));
    module.Backward(input, output, output, delta);

    REQUIRE(accu(delta) == 6336.0);
    REQUIRE(accu(output) == 416.0);
    REQUIRE(weights.n_elem == 9 + 1);
  }
}

/**
 * Test the numerical gradient of the Transposed Convolution layer.
 */
TEST_CASE("TransposedConvolutionGradientTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::linspace<arma::colvec>(0, 35, 36)),
        target(arma::mat("1"))
    {
      model = new FFN<NegativeLogLikelihood, RandomInitialization>();
      model->ResetData(input, target);
      model->Add<TransposedConvolution>(1, 3, 3, 1, 1, 0, 0, "same");
      model->Add<LogSoftMax>();
      model->InputDimensions() = std::vector<size_t>({ 6, 6 });
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) < 1e-4);
}
