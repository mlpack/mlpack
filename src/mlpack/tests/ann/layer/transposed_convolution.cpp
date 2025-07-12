
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
#include <mlpack/methods/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
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
TEST_CASE("TransposedConvolutionDimensionTest", "[ANNLayerTest]")
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
  for (auto& c : configs)
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

    // WeightSize = inMaps * maps * kernelWidth + maps (bias).
    size_t expectedWeightSize = inMaps * maps * c.kernelWidth
        * c.kernelHeight + maps;
    REQUIRE(module.WeightSize() == expectedWeightSize);

    // OutputSize = (outputWidth * outputHeight * maps).
    size_t expectedOutputSize = c.expecedOutputWidth * c.expecedOutputHeight
        * maps;
    REQUIRE(module.OutputSize() == expectedOutputSize);
  }
}

/**
 * Transposed Convolution Layer weight initialization test.
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
 * Simple Transposed Convolution layer test.
 **/
TEST_CASE("SimpleTransposedConvolutionTest", "[ANNLayerTest]")
{
  arma::mat input, output, delta;

  TransposedConvolution module1(1, 3, 3, 1, 1, 0, 0);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 15, 16);
  module1.InputDimensions() = std::vector<size_t>({ 4, 4 });
  module1.ComputeOutputDimensions();
  arma::mat weights1(module1.WeightSize(), 1);
  REQUIRE(weights1.n_elem == 9 + 1);
  weights1[0] = 1.0;
  weights1[8] = 2.0;
  module1.SetWeights(weights1);
  output.set_size(module1.OutputSize(), 1);
  module1.Forward(input, output);
  // Value calculated using tensorflow.nn.conv2d_transpose()
  REQUIRE(accu(output) == 360.0);
  // Test the backward function.
  delta.set_size(arma::size(input));
  module1.Backward(input, output, output, delta);
  // Value calculated using tensorflow.nn.conv2d()
  REQUIRE(accu(delta) == 720.0);

  TransposedConvolution module2(1, 4, 4, 1, 1, 1, 1);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module2.InputDimensions() = std::vector<size_t>({ 5, 5 });
  module2.ComputeOutputDimensions();
  arma::mat weights2(module2.WeightSize(), 1);
  REQUIRE(weights2.n_elem == 16 + 1);
  weights2[0] = 1.0;
  weights2[3] = 1.0;
  weights2[6] = 1.0;
  weights2[9] = 1.0;
  weights2[12] = 1.0;
  weights2[15] = 2.0;
  module2.SetWeights(weights2);
  output.set_size(module2.OutputSize(), 1);
  module2.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(accu(output) == 1512.0);
  // Test the backward function.
  delta.set_size(arma::size(input));
  module2.Backward(input, output, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(accu(delta) == 6504.0);

  TransposedConvolution module3(1, 3, 3, 1, 1, 1, 1);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module3.InputDimensions() = std::vector<size_t>({ 5, 5 });
  module3.ComputeOutputDimensions();
  arma::mat weights3(module3.WeightSize(), 1);
  REQUIRE(weights3.n_elem == 9 + 1);
  weights3[1] = 2.0;
  weights3[2] = 4.0;
  weights3[3] = 3.0;
  weights3[8] = 1.0;
  module3.SetWeights(weights3);
  output.set_size(module3.OutputSize(), 1);
  module3.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(accu(output) == 2370.0);
  // Test the backward function.
  delta.set_size(arma::size(input));
  module3.Backward(input, output, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(accu(delta) == 19154.0);

  TransposedConvolution module4(1, 3, 3, 1, 1, 0, 0);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module4.InputDimensions() = std::vector<size_t>({ 5, 5 });
  module4.ComputeOutputDimensions();
  arma::mat weights4(module4.WeightSize(), 1);
  REQUIRE(weights4.n_elem == 9 + 1);
  weights4[2] = 2.0;
  weights4[4] = 4.0;
  weights4[6] = 6.0;
  weights4[8] = 8.0;
  module4.SetWeights(weights4);
  output.set_size(module4.OutputSize(), 1);
  module4.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(accu(output) == 6000.0);
  // Test the backward function.
  delta.set_size(arma::size(input));
  module4.Backward(input, output, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(accu(delta) == 86208.0);

  TransposedConvolution module5(1, 3, 3, 2, 2, 0, 0);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module5.InputDimensions() = std::vector<size_t>({ 2, 2 });
  module5.ComputeOutputDimensions();
  arma::mat weights5(module5.WeightSize(), 1);
  REQUIRE(weights5.n_elem == 9 + 1);
  weights5(2) = 8.0;
  weights5(4) = 6.0;
  weights5(6) = 4.0;
  weights5(8) = 2.0;
  module5.SetWeights(weights5);
  output.set_size(module5.OutputSize(), 1);
  module5.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(accu(output) == 120.0);
  // Test the backward function.
  delta.set_size(arma::size(input));
  module5.Backward(input, output, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(accu(delta) == 960.0);

  TransposedConvolution module6(1, 3, 3, 2, 2, 1, 1);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module6.InputDimensions() = std::vector<size_t>({ 3, 3 });
  module6.ComputeOutputDimensions();
  arma::mat weights6(module6.WeightSize(), 1);
  REQUIRE(weights6.n_elem == 9 + 1);
  weights6(0) = 8.0;
  weights6(3) = 6.0;
  weights6(6) = 2.0;
  weights6(8) = 4.0;
  module6.SetWeights(weights6);
  output.set_size(module6.OutputSize(), 1);
  module6.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(accu(output) == 410.0);
  // Test the backward function.
  delta.set_size(arma::size(input));
  module6.Backward(input, output, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(accu(delta) == 4444.0);

  TransposedConvolution module7(1, 3, 3, 2, 2, 1, 1);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module7.InputDimensions() = std::vector<size_t>({ 3, 3 });
  module7.ComputeOutputDimensions();
  arma::mat weights7(module7.WeightSize(), 1);
  REQUIRE(weights7.n_elem == 9 + 1);
  weights7(0) = 8.0;
  weights7(2) = 6.0;
  weights7(4) = 2.0;
  weights7(8) = 4.0;
  module7.SetWeights(weights7);
  output.set_size(module7.OutputSize(), 1);
  module7.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(accu(output) == 416.0);
  // Test the backward function.
  delta.set_size(arma::size(input));
  module7.Backward(input, output, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(accu(delta) == 6336.0);
}

/**
 * Transposed Convolution layer numerical gradient test.
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
      model->Add<TransposedConvolution>(
          1,
          3,
          3,
          1,
          1,
          std::tuple<size_t, size_t>(0, 0),
          std::tuple<size_t, size_t>(0, 0),
          "same");
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

  REQUIRE(CheckGradient(function) < 1e-3);
}


// TODO: Uncomment when output padding has been added
// /**
//  * Transposed Convolution layer numerical gradient test with stride = 2.
//  */
// TEST_CASE("TranposedConvolutionGradientTestWithStride", "[ANNLayerTest]")
// {
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::linspace<arma::colvec>(0, 35, 36)),
//         target(arma::mat("1"))
//     {
//       model = new FFN<NegativeLogLikelihood, RandomInitialization>();
//       model->ResetData(input, target);
//       model->Add<TransposedConvolution>(
//         1,
//         3,
//         3,
//         2,
//         2,
//         std::tuple<size_t, size_t>(0, 0),
//         std::tuple<size_t, size_t>(0, 0),
//         "same");
//       model->Add<LogSoftMax>();
//       model->InputDimensions() = std::vector<size_t>({ 6, 6 });
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, RandomInitialization>* model;
//     arma::mat input, target;
//   } function;

//   REQUIRE(CheckGradient(function) < 1e-3);
// }
