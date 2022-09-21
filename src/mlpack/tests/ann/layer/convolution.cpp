/**
 * @file tests/ann/layer/convolution.cpp
 * @author Marcus Edel
 * @author Praveen Ch
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
 * Test that the functions that can modify and access the parameters of the
 * Convolution layer work.
 */
TEST_CASE("ConvolutionLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order: outSize, kW, kH, dW, dH, padW, padH, paddingType.
  Convolution layer1(2, 3, 4, 5, 6, std::tuple<size_t, size_t>(7, 8),
      std::tuple<size_t, size_t>(9, 10), "none");
  Convolution layer2(3, 4, 5, 6, 7, std::tuple<size_t, size_t>(8, 9),
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
 * Test that the padding options are working correctly in Convolution layer.
 */
TEST_CASE("ConvolutionLayerPaddingTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;

  // Check valid padding option.
  Convolution module1(1, 3, 3, 1, 1, std::tuple<size_t, size_t>(1, 1),
      std::tuple<size_t, size_t>(1, 1), "valid");
  module1.InputDimensions() = std::vector<size_t>({ 7, 7 });
  module1.ComputeOutputDimensions();
  arma::mat weights1(module1.WeightSize(), 1);
  REQUIRE(weights1.n_elem == 10);
  module1.SetWeights(weights1.memptr());

  // Test the Forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  output.set_size(module1.OutputSize(), 1);
  module1.Parameters().zeros();
  module1.Forward(input, output);

  REQUIRE(arma::accu(output) == 0);
  REQUIRE(output.n_rows == 25);
  REQUIRE(output.n_cols == 1);

  // Test the Backward function.
  delta.set_size(arma::size(input));
  module1.Backward(input, output, delta);

  // Check same padding option.
  Convolution module2(1, 3, 3, 1, 1, std::tuple<size_t, size_t>(0, 0),
      std::tuple<size_t, size_t>(0, 0), "same");
  module2.InputDimensions() = std::vector<size_t>({ 7, 7 });
  module2.ComputeOutputDimensions();
  arma::mat weights2(module2.WeightSize(), 1);
  REQUIRE(weights2.n_elem == 10);
  module2.SetWeights(weights2.memptr());

  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  output.set_size(module2.OutputSize(), 1);
  module2.Parameters().zeros();
  module2.Forward(input, output);

  REQUIRE(arma::accu(output) == 0);
  REQUIRE(output.n_rows == 49);
  REQUIRE(output.n_cols == 1);

  // Test the backward function.
  delta.set_size(arma::size(input));
  module2.Backward(input, output, delta);
}

/**
 * Convolution layer numerical gradient test.
 */
TEST_CASE("GradientConvolutionLayerTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::linspace<arma::colvec>(0, 35, 36)),
        target(arma::mat("1"))
    {
      model = new FFN<NegativeLogLikelihood, RandomInitialization>();
      model->ResetData(input, target);
      model->Add<Convolution>(1, 3, 3, 1, 1, std::tuple<size_t, size_t>(0, 0),
          std::tuple<size_t, size_t>(0, 0), "same");
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

  REQUIRE(CheckGradient(function) < 1e-1);
}

/**
 * Convolution layer numerical gradient test with stride = 2.
 */
TEST_CASE("GradientConvolutionLayerWithStrideTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::linspace<arma::colvec>(0, 35, 36)),
        target(arma::mat("1"))
    {
      model = new FFN<NegativeLogLikelihood, RandomInitialization>();
      model->ResetData(input, target);
      model->Add<Convolution>(1, 3, 3, 2, 2, std::tuple<size_t, size_t>(0, 0),
          std::tuple<size_t, size_t>(0, 0), "same");
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

  REQUIRE(CheckGradient(function) < 1e-1);
}

TEST_CASE("ConvolutionLayerTestCase", "[ANNLayerTest]")
{
  arma::mat input, output;

  // The input test matrix is of the form 3 x 2 x 4 x 1 where
  // number of images are 3 and number of feature maps are 2.
  input = { { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 } };

  Convolution layer(4, 1, 1, 1, 1, 0, 0);
  layer.InputDimensions() = std::vector<size_t>({ 4, 1, 2 });
  layer.ComputeOutputDimensions();
  arma::mat layerWeights(layer.WeightSize(), 1);
  layer.SetWeights(layerWeights.memptr());
  output.set_size(layer.OutputSize(), 3);

  // Set weights to 1.0 and bias to 0.0.
  layer.Weight().fill(1.0);
  layer.Bias().zeros();
  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  REQUIRE(arma::accu(output) == 4108);

  // Set bias to one.
  layer.Bias().fill(1.0);
  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  REQUIRE(arma::accu(output) == 4156);
}

/**
 * Convolution module weight initialization test.
 */
TEST_CASE("ConvolutionLayerWeightInitializationTest", "[ANNLayerTest]")
{
  size_t inSize = 2, outSize = 3;
  size_t kernelWidth = 4, kernelHeight = 5;
  Convolution module = Convolution(outSize,
      kernelWidth, kernelHeight, 6, 7, std::tuple<size_t, size_t>(8, 9),
      std::tuple<size_t, size_t>(10, 11), "none");
  module.InputDimensions() = std::vector<size_t>({ 12, 13, 2 });
  module.ComputeOutputDimensions();
  arma::mat weights(module.WeightSize(), 1);
  module.SetWeights(weights.memptr());

  RandomInitialization().Initialize(module.Weight());
  module.Bias().ones();

  REQUIRE(std::equal(module.Weight().begin(),
      module.Weight().end(), module.Parameters().begin()));

  REQUIRE(std::equal(module.Bias().begin(),
      module.Bias().end(), module.Parameters().end() - outSize));

  REQUIRE(module.Weight().n_rows == kernelWidth);
  REQUIRE(module.Weight().n_cols == kernelHeight);
  REQUIRE(module.Weight().n_slices == outSize * inSize);
  REQUIRE(module.Bias().n_rows == outSize);
  REQUIRE(module.Bias().n_cols == 1);
  REQUIRE(module.Parameters().n_rows
      == (outSize * inSize * kernelWidth * kernelHeight) + outSize);
}

TEST_CASE("NoBiasConvolutionLayerTestCase", "[ANNLayerTest]")
{
  arma::mat input, output;

  // The input test matrix is of the form 3 x 2 x 4 x 1 where
  // number of images are 3 and number of feature maps are 2.
  input = { { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 } };

  Convolution layer(4, 1, 1, 1, 1, 0, 0, "none", false);
  layer.InputDimensions() = std::vector<size_t>({ 4, 1, 2 });
  layer.ComputeOutputDimensions();
  REQUIRE(layer.WeightSize() == 8);
  arma::mat layerWeights(layer.WeightSize(), 1);
  layer.SetWeights(layerWeights.memptr());
  REQUIRE(layer.Bias().n_elem == 0);
  output.set_size(layer.OutputSize(), 3);

  // Set weights to 1.0 and bias to 0.0.
  layer.Weight().fill(1.0);
  layer.Bias().zeros();
  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  REQUIRE(arma::accu(output) == 4108);

  // Set bias to one.
  layer.Bias().fill(1.0);
  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  REQUIRE(arma::accu(output) == 4108);
}

/**
 * Advanced test for the Convolution layer.
 */
TEST_CASE("AdvancedConvolutionLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output;

  // The input test matrix is of the form 3 x 2 x 2 x 2 where
  // number of images are 3 and number of feature maps are 2.
  input = { { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 } };

  Convolution layer(2, 2, 2, 1, 1, 0, 0);
  layer.InputDimensions() = std::vector<size_t>({ 2, 2, 2 });
  layer.ComputeOutputDimensions();
  arma::mat layerWeights(layer.WeightSize(), 1);
  layerWeights(0) = 0.23757622;
  layerWeights(1) = -0.11899071;
  layerWeights(2) = 0.10450475;
  layerWeights(3) = -0.1303806;
  layerWeights(4) = -0.34706244;
  layerWeights(5) = -0.09472395;
  layerWeights(6) = 0.04117536;
  layerWeights(7) = -0.23012237;
  layerWeights(8) = -0.02827594;
  layerWeights(9) = -0.24280427;
  layerWeights(10) = 0.33375624;
  layerWeights(11) = -0.12285174;
  layerWeights(12) = -0.05546845;
  layerWeights(13) = -0.01502632;
  layerWeights(14) = -0.25894147;
  layerWeights(15) = -0.2283206;
  layerWeights(16) = 0.3204123974;
  layerWeights(17) = 0.2334779799;
  layer.SetWeights(layerWeights.memptr());
  output.set_size(layer.OutputSize(), 3);

  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  REQUIRE(arma::accu(output) == Approx(12.6755657196).epsilon(1e-5));

  arma::mat delta;
  delta.set_size(8, 3);
  layer.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == Approx(-1.9237523079).epsilon(1e-5));
}

/**
 * Advanced test for the Convolution layer with stride = 2.
 */
TEST_CASE("AdvancedConvolutionLayerWithStrideTest", "[ANNLayerTest]")
{
  arma::mat input, output;

  // The input test matrix is of the form 3 x 2 x 2 x 2 where
  // number of images are 3 and number of feature maps are 2.
  input = { { 1, 446, 42 },
            { 2, 16, 63 },
            { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 446, 42 },
            { 2, 16, 63 },
            { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 } };

  Convolution layer(2, 2, 2, 2, 2, 0, 0);
  layer.InputDimensions() = std::vector<size_t>({ 4, 4, 2 });
  layer.ComputeOutputDimensions();
  arma::mat layerWeights(layer.WeightSize(), 1);
  layerWeights(0) = 0.34526727;
  layerWeights(1) = 0.10398731;
  layerWeights(2) = -0.23198915;
  layerWeights(3) = 0.05350551;
  layerWeights(4) = -0.2239646;
  layerWeights(5) = 0.30852968;
  layerWeights(6) = -0.2635072;
  layerWeights(7) = 0.01724506;
  layerWeights(8) = -0.20932047;
  layerWeights(9) = 0.2990749;
  layerWeights(10) = -0.2981235;
  layerWeights(11) = -0.14024211;
  layerWeights(12) = -0.09744886;
  layerWeights(13) = 0.16249102;
  layerWeights(14) = 0.2692932;
  layerWeights(15) = -0.12563613;
  layerWeights(16) = -0.1114468053;
  layerWeights(17) = -0.3029643595;
  layer.SetWeights(layerWeights.memptr());
  output.set_size(layer.OutputSize(), 3);

  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  REQUIRE(arma::accu(output) == Approx(364.7379150391).epsilon(1e-5));

  arma::mat delta;
  delta.set_size(32, 3);
  layer.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == Approx(115.3515701294).epsilon(1e-5));
}
