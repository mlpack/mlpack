/**
 * @file tests/ann_layer_test.cpp
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

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include "../test_catch_tools.hpp"
#include "../catch.hpp"
#include "../serialization.hpp"
#include "ann_test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

/**
 * Simple dropout module test.
 */
TEST_CASE("SimpleDropoutLayerTest", "[ANNLayerTest]")
{
  // Initialize the probability of setting a value to zero.
  const double p = 0.2;

  // Initialize the input parameter.
  arma::mat input(1000, 1);
  input.fill(1 - p);

  Dropout module(p);
  module.Training() = true;

  // Test the Forward function.
  arma::mat output;
  module.Forward(input, output);
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(output) - (1 - p))) <= 0.05);

  // Test the Backward function.
  arma::mat delta;
  module.Backward(input, input, delta);
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(delta) - (1 - p))) <= 0.05);

  // Test the Forward function.
  module.Training() = false;
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == arma::accu(output));
}

/**
 * Perform dropout x times using ones as input, sum the number of ones and
 * validate that the layer is producing approximately the correct number of
 * ones.
 */
TEST_CASE("DropoutProbabilityTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      Dropout module(probability[trial]);
      module.Training() = true;

      arma::mat output;
      module.Forward(input, output);

      // Return a column vector containing the indices of elements of X that
      // are non-zero, we just need the number of non-zero values.
      arma::uvec nonzero = arma::find(output);
      nonzeroCount += nonzero.n_elem;
    }
    const double expected = input.n_elem * (1 - probability[trial]) *
        iterations;
    const double error = fabs(nonzeroCount - expected) / expected;

    REQUIRE(error <= 0.15);
  }
}

/*
 * Perform dropout with probability 1 - p where p = 0, means no dropout.
 */
TEST_CASE("NoDropoutTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  Dropout module(0);
  module.Training() = true;

  arma::mat output;
  module.Forward(input, output);

  REQUIRE(arma::accu(output) == arma::accu(input));
}

/*
 * Perform test to check whether mean and variance remain nearly same
 * after AlphaDropout.
 */
TEST_CASE("SimpleAlphaDropoutLayerTest", "[ANNLayerTest]")
{
  // Initialize the probability of setting a value to alphaDash.
  const double p = 0.2;

  // Initialize the input parameter having a mean nearabout 0
  // and variance nearabout 1.
  arma::mat input = arma::randn<arma::mat>(1000, 1);

  AlphaDropout module(p);
  module.Training() = true;

  // Test the Forward function when training phase.
  arma::mat output(arma::size(input));
  module.Forward(input, output);
  // Check whether mean remains nearly same.
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(input) - arma::mean(output))) <=
      0.1);

  // Check whether variance remains nearly same.
  REQUIRE(arma::as_scalar(arma::abs(arma::var(input) - arma::var(output))) <=
      0.1);

  // Test the Backward function when training phase.
  arma::mat delta;
  module.Backward(input, input, delta);
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(delta) - 0)) <= 0.05);

  // Test the Forward function when testing phase.
  module.Training() = false;
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == arma::accu(output));
}

/**
 * Perform AlphaDropout x times using ones as input, sum the number of ones
 * and validate that the layer is producing approximately the correct number
 * of ones.
 */
TEST_CASE("AlphaDropoutProbabilityTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      AlphaDropout module(probability[trial]);
      module.Training() = true;

      arma::mat output(arma::size(input));
      module.Forward(input, output);

      // Return a column vector containing the indices of elements of X
      // that are not alphaDash, we just need the number of
      // nonAlphaDash values.
      arma::uvec nonAlphaDash = arma::find(module.Mask());
      nonzeroCount += nonAlphaDash.n_elem;
    }

    const double expected = input.n_elem * (1-probability[trial]) * iterations;

    const double error = fabs(nonzeroCount - expected) / expected;

    REQUIRE(error <= 0.15);
  }
}

/**
 * Perform AlphaDropout with probability 1 - p where p = 0,
 * means no AlphaDropout.
 */
TEST_CASE("NoAlphaDropoutTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  AlphaDropout module(0);
  module.Training() = false;

  arma::mat output;
  module.Forward(input, output);

  REQUIRE(arma::accu(output) == arma::accu(input));
}

/**
 * Simple Linear3D layer test.
 */
TEST_CASE("SimpleLinear3DLayerTest", "[ANNLayerTest]")
{
  const size_t inSize = 4;
  const size_t outSize = 1;
  const size_t nPoints = 2;
  const size_t batchSize = 1;
  arma::mat input, output, delta;

  // Create a Linear3D layer outside of a network, and then set its memory.
  Linear3D module(outSize);
  module.InputDimensions() = std::vector<size_t>({ 4, 2 });
  module.ComputeOutputDimensions();
  arma::mat weights(module.WeightSize(), 1);
  module.SetWeights(weights.memptr());

  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(inSize * nPoints, batchSize);
  output.set_size(outSize * nPoints, batchSize);
  module.Forward(input, output);
  REQUIRE(arma::accu(module.Bias())
      == Approx(arma::accu(output) / (nPoints * batchSize)).epsilon(1e-3));

  // Test the Backward function.
  delta.set_size(input.n_rows, input.n_cols);
  output.zeros();
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0);
}

/**
 * Jacobian Linear3D module test.
 */
TEST_CASE("JacobianLinear3DLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inSize = math::RandInt(2, 10);
    const size_t outSize = math::RandInt(2, 10);
    const size_t nPoints = math::RandInt(2, 10);
    const size_t batchSize = 1;

    arma::mat input;
    input.set_size(inSize * nPoints, batchSize);

    // Create a Linear3D layer outside a network and initialize its memory.
    Linear3D module(outSize);
    module.InputDimensions() = std::vector<size_t>({ inSize, nPoints });
    module.ComputeOutputDimensions();
    arma::mat weights(module.WeightSize(), 1);
    module.SetWeights(weights.memptr());

    module.Parameters().randu();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}

/**
 * Simple Gradient test for Linear3D layer.
 */
TEST_CASE("GradientLinear3DLayerTest", "[ANNLayerTest]")
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        inSize(4),
        outSize(1),
        nPoints(2),
        batchSize(4)
    {
      input = arma::randu(inSize * nPoints, batchSize);
      target = arma::zeros(outSize * nPoints, batchSize);
      target(0, 0) = 1;
      target(0, 3) = 1;
      target(1, 1) = 1;
      target(1, 2) = 1;

      model = new FFN<MeanSquaredError, RandomInitialization>();
      model->ResetData(input, target);
      model->Add<Linear3D>(outSize);
      model->InputDimensions() = std::vector<size_t>{ 4, 2 };
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

    FFN<MeanSquaredError, RandomInitialization>* model;
    arma::mat input, target;
    const size_t inSize;
    const size_t outSize;
    const size_t nPoints;
    const size_t batchSize;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-7);
}

/**
 * Simple linear no bias module test.
 */
TEST_CASE("SimpleLinearNoBiasLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  LinearNoBias module(10);
  arma::mat weights(10 * 10, 1);
  module.InputDimensions() = std::vector<size_t>({ 10 });
  module.ComputeOutputDimensions();
  module.SetWeights(weights.memptr());

  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  REQUIRE(0 == arma::accu(output));

  // Test the Backward function.
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0);
}

/**
 * Simple padding layer test.
 */
TEST_CASE("SimplePaddingLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  Padding module(1, 2, 3, 4);
  module.InputDimensions() = std::vector<size_t>({ 2, 5 });
  module.ComputeOutputDimensions();

  // Test the Forward function.
  input = arma::randu(10, 1);
  size_t totalOutputDimensions = module.OutputDimensions()[0];
  for (size_t i = 1; i < module.OutputDimensions().size(); ++i)
    totalOutputDimensions *= module.OutputDimensions()[i];
  output.set_size(totalOutputDimensions, input.n_cols);
  output.randu();
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == Approx(arma::accu(output)));
  REQUIRE(output.n_rows == (9 * 8)); // 2x5 --> 9x8

  // Test the Backward function.
  delta.set_size(input.n_rows, input.n_cols);
  module.Backward(input, output, delta);
  CheckMatrices(delta, input);

  // Test forward function for multiple filters.
  // Here it's 3 filters with height = 224, width = 224
  // the output should be [226 * 226 * 3, 1] with 1 padding.
  module = Padding(1, 1, 1, 1);
  module.InputDimensions() = std::vector<size_t>({ 224, 224, 3 });
  module.ComputeOutputDimensions();

  input = arma::randu(224 * 224 * 3, 1);
  totalOutputDimensions = module.OutputDimensions()[0];
  for (size_t i = 1; i < module.OutputDimensions().size(); ++i)
    totalOutputDimensions *= module.OutputDimensions()[i];
  output.set_size(totalOutputDimensions, input.n_cols);
  output.randu();
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == Approx(arma::accu(output)));
  REQUIRE(output.n_rows == (226 * 226 * 3));
  REQUIRE(output.n_cols == 1);

  // Test forward function for multiple batches with multiple filters.
  // Here it's 3 filters with height = 244, width = 244
  // the output should be [246 * 246 * 3, 3] with 1 padding.
  module.InputDimensions() = std::vector<size_t>({ 244, 244, 3 });
  module.ComputeOutputDimensions();
  totalOutputDimensions = module.OutputDimensions()[0];
  for (size_t i = 1; i < module.OutputDimensions().size(); ++i)
    totalOutputDimensions *= module.OutputDimensions()[i];

  input = arma::randu(244 * 244 * 3, 3);
  output.set_size(totalOutputDimensions, input.n_cols);
  output.randu();
  module.Forward(input, output);
  REQUIRE(output.n_rows == (246 * 246 * 3));
  REQUIRE(output.n_cols == 3);
  REQUIRE(arma::accu(input) == Approx(arma::accu(output)));
}

/**
 * Jacobian linear no bias module test.
 */
TEST_CASE("JacobianLinearNoBiasLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = math::RandInt(2, 1000);
    const size_t outputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    LinearNoBias module(outputElements);
    arma::mat weights(inputElements * outputElements, 1);
    module.InputDimensions() = std::vector<size_t>({ inputElements });
    module.ComputeOutputDimensions();
    module.SetWeights(weights.memptr());

    module.Parameters().randu();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}

/**
 * LinearNoBias layer numerical gradient test.
 */
TEST_CASE("GradientLinearNoBiasLayerTest", "[ANNLayerTest]")
{
  // LinearNoBias function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(10);
      model->Add<LinearNoBias>(2);
      model->Add<LogSoftMax>();
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

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

/**
 * Simple LogSoftMax module test.
 */
TEST_CASE("SimpleLogSoftmaxLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, error, delta;
  LogSoftMax module;

  // Test the Forward function.
  input = arma::mat("0.5; 0.5");
  module.Forward(input, output);
  REQUIRE(arma::accu(arma::abs(arma::mat("-0.6931; -0.6931") - output)) ==
      Approx(0.0).margin(1e-3));

  // Test the Backward function.
  error = arma::zeros(input.n_rows, input.n_cols);
  // Assume LogSoftmax layer is always associated with NLL output layer.
  error(1, 0) = -1;
  module.Backward(input, error, delta);
  REQUIRE(arma::accu(arma::abs(arma::mat("1.6487; 0.6487") - delta)) ==
      Approx(0.0).margin(1e-3));
}

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

  REQUIRE(CheckGradient(function) < 1e3);
}

/**
 * Simple test for Max Pooling layer.
 */
TEST_CASE("MaxPoolingTestCase", "[ANNLayerTest]")
{
  // For rectangular input to pooling layers.
  arma::mat input = arma::mat(12, 1);
  arma::mat output;
  input.zeros();
  input(0) = 1;
  input(1) = 2;
  input(2) = 3;
  input(3) = input(8) = 7;
  input(4) = 4;
  input(5) = 5;
  input(6) = input(7) = 6;
  input(10) = 8;
  input(11) = 9;
  // Output-Size should be 2 x 2.
  output.set_size(4, 1);

  // Square output.
  MaxPooling module1(2, 2, 2, 1);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  module1.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  REQUIRE(arma::accu(output) == 28);
  REQUIRE(output.n_elem == 4);
  REQUIRE(output.n_cols == 1);

  // For Square input.
  input = arma::mat(9, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(3) = 3;
  input(6) = 3;
  // Output-Size should be 1 x 2.
  output.set_size(2, 1);

  // Rectangular output.
  MaxPooling module2(3, 2, 3, 1);
  module2.InputDimensions() = std::vector<size_t>({ 3, 3 });
  module2.ComputeOutputDimensions();
  module2.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  REQUIRE(arma::accu(output) == 12.0);
  REQUIRE(output.n_elem == 2);
  REQUIRE(output.n_cols == 1);

  // For Square input.
  input = arma::mat(16, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(4) = 3;
  input(8) = 3;
  // Output-Size should be 3 x 3.
  output.set_size(9, 1);

  // Square output.
  MaxPooling module3(2, 2, 1, 1);
  module3.InputDimensions() = std::vector<size_t>({ 4, 4 });
  module3.ComputeOutputDimensions();
  module3.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  REQUIRE(arma::accu(output) == 30.0);
  REQUIRE(output.n_elem == 9);
  REQUIRE(output.n_cols == 1);

  // For Rectangular input.
  input = arma::mat(6, 1);
  input.zeros();
  input(0) = 1;
  input(1) = 1;
  input(3) = 1;
  // Output-Size should be 2 x 2.
  output.set_size(4, 1);

  // Square output.
  MaxPooling module4(2, 1, 1, 1);
  module4.InputDimensions() = std::vector<size_t>({ 3, 2 });
  module4.ComputeOutputDimensions();
  module4.Forward(input, output);
  // Calculated using torch.nn.MaxPool2d().
  REQUIRE(arma::accu(output) == 3);
  REQUIRE(output.n_elem == 4);
  REQUIRE(output.n_cols == 1);
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
