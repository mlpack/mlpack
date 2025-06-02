/**
 * @file tests/ann/layer/linear_recurrent.cpp
 * @author Ryan Curtin
 *
 * Tests the LinearRecurrent layer.
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
 * Simple LinearRecurrent layer test.
 */
TEST_CASE("SimpleLinearRecurrentLayerTest", "[ANNLayerTest]")
{
  const size_t inSize = 4;
  const size_t outSize = 1;
  const size_t batchSize = 2;
  arma::mat input, output, delta;

  // Create a LinearRecurrent layer outside of a network, and then set its
  // memory.
  LinearRecurrent module(outSize);
  module.InputDimensions() = std::vector<size_t>({ 4 });
  module.ComputeOutputDimensions();
  arma::mat weights(module.WeightSize(), 1);
  module.SetWeights(weights);
  module.ClearRecurrentState(1, batchSize);
  module.CurrentStep(0, true /* only one step */);

  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(inSize, batchSize);
  output.set_size(outSize, batchSize);
  module.Forward(input, output);
  REQUIRE(accu(module.Bias())
      == Approx(accu(output) / (batchSize)).epsilon(1e-3));

  // Test the Backward function.
  delta.set_size(input.n_rows, input.n_cols);
  output.zeros();
  module.Backward(input, output, output, delta);
  REQUIRE(accu(delta) == 0);
}

/**
 * Jacobian LinearRecurrent module test.
 */
TEST_CASE("JacobianLinearRecurrentLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inSize = RandInt(2, 10);
    const size_t outSize = RandInt(2, 10);
    const size_t batchSize = 1;

    arma::mat input;
    input.set_size(inSize, batchSize);

    // Create a LinearRecurrent layer outside a network and initialize its
    // memory.
    LinearRecurrent module(outSize);
    module.InputDimensions() = std::vector<size_t>({ inSize });
    module.ComputeOutputDimensions();
    arma::mat weights(module.WeightSize(), 1);
    module.SetWeights(weights);
    module.ClearRecurrentState(1, batchSize);
    module.CurrentStep(0, true /* only one step */);

    module.Parameters().randu();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}

/**
 * Simple Gradient test for LinearRecurrent layer.
 */
TEST_CASE("GradientLinearRecurrentLayerTest", "[ANNLayerTest]")
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        inSize(15),
        outSize(5),
        batchSize(32)
    {
      input = arma::randu(inSize, batchSize, 4);
      target = arma::zeros(outSize, batchSize, 4);
      target(0, 0, 0) = 1;
      target(0, 3, 0) = 1;
      target(0, 5, 0) = 1;
      target(0, 0, 1) = 7;
      target(0, 3, 1) = 5;
      target(0, 5, 1) = 3;
      target(0, 1, 2) = 7;
      target(0, 4, 2) = 5;
      target(0, 2, 3) = 7;
      target(0, 4, 3) = 5;

      model = RNN<MeanSquaredError, RandomInitialization>(4);
      model.ResetData(input, target);
      model.Add<LinearRecurrent>(outSize);
      model.InputDimensions() = std::vector<size_t>{ 15 };
    }

    double Gradient(arma::mat& gradient)
    {
      return model.EvaluateWithGradient(model.Parameters(), 0, gradient, 1);
    }

    arma::mat& Parameters() { return model.Parameters(); }

    RNN<MeanSquaredError, RandomInitialization> model;
    arma::cube input, target;
    const size_t inSize;
    const size_t outSize;
    const size_t batchSize;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-7);
}

/**
 * Test the recurrent functionality of the LinearRecurrent layer with a very
 * simple manual test.
 */
TEST_CASE("TrivialLinearRecurrentLayerRecurrentTest", "[ANNLayerTest]")
{
  // To ensure that the forward pass properly accounts for hidden state, we'll
  // do a simple test: set all the weight matrices to the identity matrix, and
  // the biases to 0.  At the first time step, pass in a 0 for each dimension.
  // We'll expect a 0 for each output.
  //
  // At the next time step, we'll pass in a 1.  Now we expect 1 at each output.
  //
  // Then, we'll pass in 0 for 10 time steps, expecting 1+i at each output for
  // time step i (of 10).
  const size_t size = 25;
  const size_t batchSize = 3;
  arma::mat input, output;

  // Create a LinearRecurrent layer outside of a network, and then set its
  // memory.
  LinearRecurrent l(size);
  l.InputDimensions() = std::vector<size_t>({ size });
  l.ComputeOutputDimensions();
  arma::mat weights(l.WeightSize(), 1);
  l.SetWeights(weights);

  // Set the weight values to what we need for the test.
  l.Weights().eye();
  l.RecurrentWeights().eye();
  l.Bias().zeros();

  l.ClearRecurrentState(1, batchSize);
  l.CurrentStep(0);

  input.zeros(size, batchSize);
  output.ones(size, batchSize);

  // Pass zeros through; we expect zero outputs.
  l.Forward(input, output);

  REQUIRE(all(all(output == 0.0)));

  // Now pass ones through.
  l.CurrentStep(1);
  input.ones();
  l.Forward(input, output);

  REQUIRE(all(all(output == 1.0)));

  // Now take ten steps where we pass nothing through, but expect the same
  // output each time.
  input.zeros();
  for (size_t t = 2; t < 12; ++t)
  {
    l.CurrentStep(t);
    l.Forward(input, output);

    REQUIRE(all(all(output == 1.0)));
  }
}

/**
 * Test that Backward() properly accounts for the recurrent state gradient.
 */
TEST_CASE("LinearRecurrentBackwardStateTest", "[ANNLayerTest]")
{
  // The recurrent state gradient should produce equivalent results to the
  // regular gradient, because the recurrent connection is to the output of the
  // layer.
  //
  // So, we will create a series of random inputs and states, and then ensure
  // that the computed delta from Backward() is the same when passing dawta
  // through the recurrent state connection or through the output.
  const size_t size = 5;
  const size_t batchSize = 2;
  arma::mat input, output, error;

  // Create a LinearRecurrent layer outside of a network, and then set its
  // memory.
  LinearRecurrent l(size);
  l.InputDimensions() = std::vector<size_t>({ size });
  l.ComputeOutputDimensions();
  arma::mat weights(l.WeightSize(), 1);
  l.SetWeights(weights);
  l.Parameters().randu(); // just initialize parameters randomly
  l.ClearRecurrentState(1 /* memory size of 1 */, batchSize);

  for (size_t trial = 0; trial < 10; ++trial)
  {
    arma::mat deltaOutput, deltaRecurrent, deltaBoth, error;

    // We set the time step to 0, but say it is not the end of the sequence.
    // This is okay because we will be manually setting and modifying the
    // recurrent state.
    l.CurrentStep(0);

    // Create a random input and pass it through the network.
    input.randu(size, batchSize);
    l.Forward(input, output);

    error.randu(output.n_rows, output.n_cols);
    deltaOutput.set_size(output.n_rows, output.n_cols);
    deltaRecurrent.set_size(output.n_rows, output.n_cols);
    deltaBoth.set_size(output.n_rows, output.n_cols);

    // First compute the delta when we are at the last time step, so that the
    // recurrent state connection contributes nothing.
    l.CurrentStep(1, true /* end of sequence */);
    l.Backward(input, output, error, deltaOutput);

    // Now compute the delta when we set the output error to 0, so all error
    // goes through the recurrent state.  Manually modify the recurrent gradient
    // state to the error.  (This testing trick is specific to this layer type!)
    l.CurrentStep(0);
    l.RecurrentGradient(0) = error;
    error.zeros();
    l.Backward(input, output, error, deltaRecurrent);

    REQUIRE(arma::approx_equal(deltaOutput, deltaRecurrent, "both", 1e-5,
        1e-5));

    // Now, if we pass both things through, we should get twice the output.
    error = l.RecurrentGradient(1);
    l.Backward(input, output, error, deltaBoth);

    REQUIRE(arma::approx_equal(deltaBoth, 2 * deltaOutput, "both", 1e-5, 1e-5));
  }
}

/**
 * Test that Gradient() properly accounts for the recurrent state gradient.
 */
TEST_CASE("LinearRecurrentGradientStateTest", "[ANNLayerTest]")
{
  const size_t size = 5;
  const size_t batchSize = 2;
  arma::mat input, output, error;

  // Create a LinearRecurrent layer outside of a network, and then set its
  // memory.
  LinearRecurrent l(size);
  l.InputDimensions() = std::vector<size_t>({ size });
  l.ComputeOutputDimensions();
  arma::mat weights(l.WeightSize(), 1);
  l.SetWeights(weights);
  l.Parameters().randu(); // just initialize parameters randomly
  l.ClearRecurrentState(3 /* memory size */, batchSize);

  for (size_t trial = 0; trial < 10; ++trial)
  {
    arma::mat gradientOutput, gradientRecurrent, gradientBoth, error;

    // We set the time step to 0, but say it is not the end of the sequence.
    // This is okay because we will be manually setting and modifying the
    // recurrent state.
    l.CurrentStep(1);

    // Create a random input and error to compute the gradient with.
    input.randu(size, batchSize);
    error.randu(size, batchSize);
    l.RecurrentState(0) = arma::randu<arma::mat>(size, batchSize);
    l.RecurrentState(1) = arma::randu<arma::mat>(size, batchSize);

    gradientOutput.set_size(l.WeightSize(), 1);
    gradientRecurrent.set_size(l.WeightSize(), 1);
    gradientBoth.set_size(l.WeightSize(), 1);

    // First compute the gradient when we are at the last time step, so that the
    // recurrent state connection contributes nothing.
    l.CurrentStep(2, true /* end of sequence */);
    l.Gradient(input, error, gradientOutput);

    // Now compute the delta when we set the output error to 0, so all error
    // goes through the recurrent state.  Manually modify the recurrent gradient
    // state to the error.  (This testing trick is specific to this layer type!)
    l.CurrentStep(2, false /* not end of sequence */);
    l.RecurrentGradient(2) = error;
    error.zeros();
    l.Gradient(input, error, gradientRecurrent);

    REQUIRE(arma::approx_equal(gradientOutput, gradientRecurrent, "both", 1e-5,
        1e-5));

    // Now, if we pass both things through, we should get twice the output.
    error = l.RecurrentGradient(2);
    l.Gradient(input, error, gradientBoth);

    REQUIRE(arma::approx_equal(gradientBoth, 2 * gradientOutput, "both", 1e-5,
        1e-5));
  }
}
