/**
 * @file ann_layer_test.cpp
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
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "ann_test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNLayerTest);

/**
 * Simple add module test.
 */
BOOST_AUTO_TEST_CASE(SimpleAddLayerTest)
{
  arma::mat output, input, delta;
  Add<> module(10);
  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(module.Parameters()), arma::accu(output));

  // Test the Backward function.
  module.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(delta));

  // Test the forward function.
  input = arma::ones(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_CLOSE(10 + arma::accu(module.Parameters()),
      arma::accu(output), 1e-3);

  // Test the backward function.
  module.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_CLOSE(arma::accu(output), arma::accu(delta), 1e-3);
}

/**
 * Jacobian add module test.
 */
BOOST_AUTO_TEST_CASE(JacobianAddLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t elements = math::RandInt(2, 1000);
    arma::mat input;
    input.set_size(elements, 1);

    Add<> module(elements);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Add layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientAddLayerTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<Add<> >(10);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple constant module test.
 */
BOOST_AUTO_TEST_CASE(SimpleConstantLayerTest)
{
  arma::mat output, input, delta;
  Constant<> module(10, 3.0);

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(output), 30.0);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);

  // Test the forward function.
  input = arma::ones(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(output), 30.0);

  // Test the backward function.
  module.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Jacobian constant module test.
 */
BOOST_AUTO_TEST_CASE(JacobianConstantLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t elements = math::RandInt(2, 1000);
    arma::mat input;
    input.set_size(elements, 1);

    Constant<> module(elements, 1.0);

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Simple dropout module test.
 */
BOOST_AUTO_TEST_CASE(SimpleDropoutLayerTest)
{
  // Initialize the probability of setting a value to zero.
  const double p = 0.2;

  // Initialize the input parameter.
  arma::mat input(1000, 1);
  input.fill(1 - p);

  Dropout<> module(p);
  module.Deterministic() = false;

  // Test the Forward function.
  arma::mat output;
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(output) - (1 - p))), 0.05);

  // Test the Backward function.
  arma::mat delta;
  module.Backward(std::move(input), std::move(input), std::move(delta));
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(delta) - (1 - p))), 0.05);

  // Test the Forward function.
  module.Deterministic() = true;
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(output));
}

/**
 * Perform dropout x times using ones as input, sum the number of ones and
 * validate that the layer is producing approximately the correct number of
 * ones.
 */
BOOST_AUTO_TEST_CASE(DropoutProbabilityTest)
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      Dropout<> module(probability[trial]);
      module.Deterministic() = false;

      arma::mat output;
      module.Forward(std::move(input), std::move(output));

      // Return a column vector containing the indices of elements of X that
      // are non-zero, we just need the number of non-zero values.
      arma::uvec nonzero = arma::find(output);
      nonzeroCount += nonzero.n_elem;
    }
    const double expected = input.n_elem * (1 - probability[trial]) *
        iterations;
    const double error = fabs(nonzeroCount - expected) / expected;

    BOOST_REQUIRE_LE(error, 0.15);
  }
}

/*
 * Perform dropout with probability 1 - p where p = 0, means no dropout.
 */
BOOST_AUTO_TEST_CASE(NoDropoutTest)
{
  arma::mat input = arma::ones(1500, 1);
  Dropout<> module(0);
  module.Deterministic() = false;

  arma::mat output;
  module.Forward(std::move(input), std::move(output));

  BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(input));
}

/*
 * Perform test to check whether mean and variance remain nearly same
 * after AlphaDropout.
 */
BOOST_AUTO_TEST_CASE(SimpleAlphaDropoutLayerTest)
{
  // Initialize the probability of setting a value to alphaDash.
  const double p = 0.2;

  // Initialize the input parameter having a mean nearabout 0
  // and variance nearabout 1.
  arma::mat input = arma::randn<arma::mat>(1000, 1);

  AlphaDropout<> module(p);
  module.Deterministic() = false;

  // Test the Forward function when training phase.
  arma::mat output;
  module.Forward(std::move(input), std::move(output));
  // Check whether mean remains nearly same.
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(input) - arma::mean(output))), 0.1);

  // Check whether variance remains nearly same.
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::var(input) - arma::var(output))), 0.1);

  // Test the Backward function when training phase.
  arma::mat delta;
  module.Backward(std::move(input), std::move(input), std::move(delta));
  BOOST_REQUIRE_LE(
      arma::as_scalar(arma::abs(arma::mean(delta) - 0)), 0.05);

  // Test the Forward function when testing phase.
  module.Deterministic() = true;
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(output));
}

/**
 * Perform AlphaDropout x times using ones as input, sum the number of ones
 * and validate that the layer is producing approximately the correct number
 * of ones.
 */
BOOST_AUTO_TEST_CASE(AlphaDropoutProbabilityTest)
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      AlphaDropout<> module(probability[trial]);
      module.Deterministic() = false;

      arma::mat output;
      module.Forward(std::move(input), std::move(output));

      // Return a column vector containing the indices of elements of X
      // that are not alphaDash, we just need the number of
      // nonAlphaDash values.
      arma::uvec nonAlphaDash = arma::find(module.Mask());
      nonzeroCount += nonAlphaDash.n_elem;
    }

    const double expected = input.n_elem * (1-probability[trial]) * iterations;

    const double error = fabs(nonzeroCount - expected) / expected;

    BOOST_REQUIRE_LE(error, 0.15);
  }
}

/**
 * Perform AlphaDropout with probability 1 - p where p = 0,
 * means no AlphaDropout.
 */
BOOST_AUTO_TEST_CASE(NoAlphaDropoutTest)
{
  arma::mat input = arma::ones(1500, 1);
  AlphaDropout<> module(0);
  module.Deterministic() = false;

  arma::mat output;
  module.Forward(std::move(input), std::move(output));

  BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(input));
}

/**
 * Simple linear module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLinearLayerTest)
{
  arma::mat output, input, delta;
  Linear<> module(10, 10);
  module.Parameters().randu();
  module.Reset();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_CLOSE(arma::accu(
      module.Parameters().submat(100, 0, module.Parameters().n_elem - 1, 0)),
      arma::accu(output), 1e-3);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(input), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Jacobian linear module test.
 */
BOOST_AUTO_TEST_CASE(JacobianLinearLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t inputElements = math::RandInt(2, 1000);
    const size_t outputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    Linear<> module(inputElements, outputElements);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Linear layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLinearLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple linear no bias module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLinearNoBiasLayerTest)
{
  arma::mat output, input, delta;
  LinearNoBias<> module(10, 10);
  module.Parameters().randu();
  module.Reset();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(0, arma::accu(output));

  // Test the Backward function.
  module.Backward(std::move(input), std::move(input), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Simple padding layer test.
 */
BOOST_AUTO_TEST_CASE(SimplePaddingLayerTest)
{
  arma::mat output, input, delta;
  Padding<> module(1, 2, 3, 4);

  // Test the Forward function.
  input = arma::randu(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(output));
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows + 3);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols + 7);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(output), std::move(delta));
  CheckMatrices(delta, input);
}

/**
 * Jacobian linear no bias module test.
 */
BOOST_AUTO_TEST_CASE(JacobianLinearNoBiasLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t inputElements = math::RandInt(2, 1000);
    const size_t outputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    LinearNoBias<> module(inputElements, outputElements);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * LinearNoBias layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLinearNoBiasLayerTest)
{
  // LinearNoBias function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<LinearNoBias<> >(10, 2);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Jacobian negative log likelihood module test.
 */
BOOST_AUTO_TEST_CASE(JacobianNegativeLogLikelihoodLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    NegativeLogLikelihood<> module;
    const size_t inputElements = math::RandInt(5, 100);
    arma::mat input;
    RandomInitialization init(0, 1);
    init.Initialize(input, inputElements, 1);

    arma::mat target(1, 1);
    target(0) = math::RandInt(1, inputElements - 1);

    double error = JacobianPerformanceTest(module, input, target);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Jacobian LeakyReLU module test.
 */
BOOST_AUTO_TEST_CASE(JacobianLeakyReLULayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    LeakyReLU<> module;

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Jacobian FlexibleReLU module test.
 */
BOOST_AUTO_TEST_CASE(JacobianFlexibleReLULayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    FlexibleReLU<> module;

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Flexible ReLU layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientFlexibleReLULayerTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(2, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, RandomInitialization>(
          NegativeLogLikelihood<>(), RandomInitialization(0.1, 0.5));

      model->Predictors() = input;
      model->Responses() = target;
      model->Add<Linear<> >(2, 2);
      model->Add<LinearNoBias<> >(2, 5);
      model->Add<FlexibleReLU<> >(0.05);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Jacobian MultiplyConstant module test.
 */
BOOST_AUTO_TEST_CASE(JacobianMultiplyConstantLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    MultiplyConstant<> module(3.0);

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Jacobian HardTanH module test.
 */
BOOST_AUTO_TEST_CASE(JacobianHardTanHLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t inputElements = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    HardTanH<> module;

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Simple select module test.
 */
BOOST_AUTO_TEST_CASE(SimpleSelectLayerTest)
{
  arma::mat outputA, outputB, input, delta;

  input = arma::ones(10, 5);
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    input.col(i) *= i;
  }

  // Test the Forward function.
  Select<> moduleA(3);
  moduleA.Forward(std::move(input), std::move(outputA));
  BOOST_REQUIRE_EQUAL(30, arma::accu(outputA));

  // Test the Forward function.
  Select<> moduleB(3, 5);
  moduleB.Forward(std::move(input), std::move(outputB));
  BOOST_REQUIRE_EQUAL(15, arma::accu(outputB));

  // Test the Backward function.
  moduleA.Backward(std::move(input), std::move(outputA), std::move(delta));
  BOOST_REQUIRE_EQUAL(30, arma::accu(delta));

  // Test the Backward function.
  moduleB.Backward(std::move(input), std::move(outputA), std::move(delta));
  BOOST_REQUIRE_EQUAL(15, arma::accu(delta));
}

/**
 * Simple join module test.
 */
BOOST_AUTO_TEST_CASE(SimpleJoinLayerTest)
{
  arma::mat output, input, delta;
  input = arma::ones(10, 5);

  // Test the Forward function.
  Join<> module;
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(50, arma::accu(output));

  bool b = output.n_rows == 1 || output.n_cols == 1;
  BOOST_REQUIRE_EQUAL(b, true);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(50, arma::accu(delta));

  b = delta.n_rows == input.n_rows && input.n_cols;
  BOOST_REQUIRE_EQUAL(b, true);
}

/**
 * Simple add merge module test.
 */
BOOST_AUTO_TEST_CASE(SimpleAddMergeLayerTest)
{
  arma::mat output, input, delta;
  input = arma::ones(10, 1);

  for (size_t i = 0; i < 5; ++i)
  {
    AddMerge<> module(false, false);
    const size_t numMergeModules = math::RandInt(2, 10);
    for (size_t m = 0; m < numMergeModules; ++m)
    {
      IdentityLayer<> identityLayer;
      identityLayer.Forward(std::move(input),
          std::move(identityLayer.OutputParameter()));

      module.Add<IdentityLayer<> >(identityLayer);
    }

    // Test the Forward function.
    module.Forward(std::move(input), std::move(output));
    BOOST_REQUIRE_EQUAL(10 * numMergeModules, arma::accu(output));

    // Test the Backward function.
    module.Backward(std::move(input), std::move(output), std::move(delta));
    BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(delta));
  }
}

/**
 * Test the LSTM layer with a user defined rho parameter and without.
 */
BOOST_AUTO_TEST_CASE(LSTMRrhoTest)
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> modelA(
      rho, false, NegativeLogLikelihood<>(), init);
  modelA.Add<IdentityLayer<> >();
  modelA.Add<Linear<> >(1, 10);

  // Use LSTM layer with rho.
  modelA.Add<LSTM<> >(10, 3, rho);
  modelA.Add<LogSoftMax<> >();

  // Create model without user defined rho parameter.
  RNN<NegativeLogLikelihood<> > modelB(
      rho, false, NegativeLogLikelihood<>(), init);
  modelB.Add<IdentityLayer<> >();
  modelB.Add<Linear<> >(1, 10);

  // Use LSTM layer with rho = MAXSIZE.
  modelB.Add<LSTM<> >(10, 3);
  modelB.Add<LogSoftMax<> >();

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  modelA.Train(input, target, opt);
  modelB.Train(input, target, opt);

  CheckMatrices(modelB.Parameters(), modelA.Parameters());
}

/**
 * LSTM layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLSTMLayerTest)
{
  // LSTM function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(1, 1, 5);
      target.ones(1, 1, 5);
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(rho);
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(1, 10);
      model->Add<LSTM<> >(10, 3, rho);
      model->Add<LogSoftMax<> >();
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

    RNN<NegativeLogLikelihood<> >* model;
    arma::cube input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test the FastLSTM layer with a user defined rho parameter and without.
 */
BOOST_AUTO_TEST_CASE(FastLSTMRrhoTest)
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> modelA(
      rho, false, NegativeLogLikelihood<>(), init);
  modelA.Add<IdentityLayer<> >();
  modelA.Add<Linear<> >(1, 10);

  // Use FastLSTM layer with rho.
  modelA.Add<FastLSTM<> >(10, 3, rho);
  modelA.Add<LogSoftMax<> >();

  // Create model without user defined rho parameter.
  RNN<NegativeLogLikelihood<> > modelB(
      rho, false, NegativeLogLikelihood<>(), init);
  modelB.Add<IdentityLayer<> >();
  modelB.Add<Linear<> >(1, 10);

  // Use FastLSTM layer with rho = MAXSIZE.
  modelB.Add<FastLSTM<> >(10, 3);
  modelB.Add<LogSoftMax<> >();

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  modelA.Train(input, target, opt);
  modelB.Train(input, target, opt);

  CheckMatrices(modelB.Parameters(), modelA.Parameters());
}

/**
 * FastLSTM layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientFastLSTMLayerTest)
{
  // Fast LSTM function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(1, 1, 5);
      target = arma::ones(1, 1, 5);
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(rho);
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(1, 10);
      model->Add<FastLSTM<> >(10, 3, rho);
      model->Add<LogSoftMax<> >();
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

    RNN<NegativeLogLikelihood<> >* model;
    arma::cube input, target;
  } function;

  // The threshold should be << 0.1 but since the Fast LSTM layer uses an
  // approximation of the sigmoid function the estimated gradient is not
  // correct.
  BOOST_REQUIRE_LE(CheckGradient(function), 0.2);
}

/**
 * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
 * state. Besides output, the overloaded function provides read access to cell
 * state of the LSTM layer.
 */
BOOST_AUTO_TEST_CASE(ReadCellStateParamLSTMLayerTest)
{
  const size_t rho = 5, inputSize = 3, outputSize = 2;

  // Provide input of all ones.
  arma::cube input = arma::ones(inputSize, outputSize, rho);

  arma::mat inputGate, forgetGate, outputGate, hidden;
  arma::mat outLstm, cellLstm;

  // LSTM layer.
  LSTM<> lstm(inputSize, outputSize, rho);
  lstm.Reset();
  lstm.ResetCell(rho);

  // Initialize the weights to all ones.
  lstm.Parameters().ones();

  arma::mat inputWeight = arma::ones(outputSize, inputSize);
  arma::mat outputWeight = arma::ones(outputSize, outputSize);
  arma::mat bias = arma::ones(outputSize, input.n_cols);
  arma::mat cellCalc = arma::zeros(outputSize, input.n_cols);
  arma::mat outCalc = arma::zeros(outputSize, input.n_cols);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
      // Wrap a matrix around our data to avoid a copy.
      arma::mat stepData(input.slice(seqNum).memptr(),
          input.n_rows, input.n_cols, false, true);

      // Apply Forward() on LSTM layer.
      lstm.Forward(std::move(stepData), // Input.
                   std::move(outLstm),  // Output.
                   std::move(cellLstm), // Cell state.
                   false); // Don't write into the cell state.

      // Compute the value of cell state and output.
      // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // z = tanh(W.dot(x) + W.dot(h) + b).
      hidden = arma::tanh(inputWeight * stepData +
                     outputWeight * outCalc + bias);

      // c = f * c + i * z.
      cellCalc = forgetGate % cellCalc + inputGate % hidden;

      // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // h = o * tanh(c).
      outCalc = outputGate % arma::tanh(cellCalc);

      CheckMatrices(outLstm, outCalc, 1e-12);
      CheckMatrices(cellLstm, cellCalc, 1e-12);
  }
}

/**
 * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
 * state. Besides output, the overloaded function provides write access to cell
 * state of the LSTM layer.
 */
BOOST_AUTO_TEST_CASE(WriteCellStateParamLSTMLayerTest)
{
  const size_t rho = 5, inputSize = 3, outputSize = 2;

  // Provide input of all ones.
  arma::cube input = arma::ones(inputSize, outputSize, rho);

  arma::mat inputGate, forgetGate, outputGate, hidden;
  arma::mat outLstm, cellLstm;
  arma::mat cellCalc;

  // LSTM layer.
  LSTM<> lstm(inputSize, outputSize, rho);
  lstm.Reset();
  lstm.ResetCell(rho);

  // Initialize the weights to all ones.
  lstm.Parameters().ones();

  arma::mat inputWeight = arma::ones(outputSize, inputSize);
  arma::mat outputWeight = arma::ones(outputSize, outputSize);
  arma::mat bias = arma::ones(outputSize, input.n_cols);
  arma::mat outCalc = arma::zeros(outputSize, input.n_cols);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
      // Wrap a matrix around our data to avoid a copy.
      arma::mat stepData(input.slice(seqNum).memptr(),
          input.n_rows, input.n_cols, false, true);

      if (cellLstm.is_empty())
      {
        // Set the cell state to zeros.
        cellLstm = arma::zeros(outputSize, input.n_cols);
        cellCalc = arma::zeros(outputSize, input.n_cols);
      }
      else
      {
        // Set the cell state to zeros.
        cellLstm = arma::zeros(cellLstm.n_rows, cellLstm.n_cols);
        cellCalc = arma::zeros(cellCalc.n_rows, cellCalc.n_cols);
      }

      // Apply Forward() on the LSTM layer.
      lstm.Forward(std::move(stepData), // Input.
                   std::move(outLstm),  // Output.
                   std::move(cellLstm), // Cell state.
                   true);  // Write into cell state.

      // Compute the value of cell state and output.
      // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // z = tanh(W.dot(x) + W.dot(h) + b).
      hidden = arma::tanh(inputWeight * stepData +
                     outputWeight * outCalc + bias);

      // c = f * c + i * z.
      cellCalc = forgetGate % cellCalc + inputGate % hidden;

      // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
      outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
          outputWeight * outCalc + outputWeight % cellCalc + bias)));

      // h = o * tanh(c).
      outCalc = outputGate % arma::tanh(cellCalc);

      CheckMatrices(outLstm, outCalc, 1e-12);
      CheckMatrices(cellLstm, cellCalc, 1e-12);
  }

  // Attempting to write empty matrix into cell state.
  lstm.Reset();
  lstm.ResetCell(rho);
  arma::mat stepData(input.slice(0).memptr(),
      input.n_rows, input.n_cols, false, true);

  lstm.Forward(std::move(stepData), // Input.
                   std::move(outLstm),  // Output.
                   std::move(cellLstm), // Cell state.
                   true); // Write into cell state.

  for (size_t seqNum = 1; seqNum < rho; ++seqNum)
  {
    arma::mat empty;
    // Should throw error.
    BOOST_REQUIRE_THROW(lstm.Forward(std::move(stepData), // Input.
                                     std::move(outLstm),  // Output.
                                     std::move(empty), // Cell state.
                                     true),  // Write into cell state.
                                     std::runtime_error);
  }
}

/**
 * Check if the gradients computed by GRU cell are close enough to the
 * approximation of the gradients.
 */
BOOST_AUTO_TEST_CASE(GradientGRULayerTest)
{
  // GRU function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(1, 1, 5);
      target = arma::ones(1, 1, 5);
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(rho);
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(1, 10);
      model->Add<GRU<> >(10, 3, rho);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      arma::mat output;
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    RNN<NegativeLogLikelihood<> >* model;
    arma::cube input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * GRU layer manual forward test.
 */
BOOST_AUTO_TEST_CASE(ForwardGRULayerTest)
{
  GRU<> gru(3, 3, 5);

  // Initialize the weights to all ones.
  NetworkInitialization<ConstInitialization>
    networkInit(ConstInitialization(1));
  networkInit.Initialize(gru.Model(), gru.Parameters());

  // Provide input of all ones.
  arma::mat input = arma::ones(3, 1);
  arma::mat output;

  gru.Forward(std::move(input), std::move(output));

  // Compute the z_t gate output.
  arma::mat expectedOutput = arma::ones(3, 1);
  expectedOutput *= -4;
  expectedOutput = arma::exp(expectedOutput);
  expectedOutput = arma::ones(3, 1) / (arma::ones(3, 1) + expectedOutput);
  expectedOutput = (arma::ones(3, 1)  - expectedOutput) % expectedOutput;

  // For the first input the output should be equal to the output of
  // gate z_t as the previous output fed to the cell is all zeros.
  BOOST_REQUIRE_LE(arma::as_scalar(arma::trans(output) * expectedOutput), 1e-2);

  expectedOutput = output;

  gru.Forward(std::move(input), std::move(output));

  double s = arma::as_scalar(arma::sum(expectedOutput));

  // Compute the value of z_t gate for the second input.
  arma::mat z_t = arma::ones(3, 1);
  z_t *= -(s + 4);
  z_t = arma::exp(z_t);
  z_t = arma::ones(3, 1) / (arma::ones(3, 1) + z_t);

  // Compute the value of o_t gate for the second input.
  arma::mat o_t = arma::ones(3, 1);
  o_t *= -(arma::as_scalar(arma::sum(expectedOutput % z_t)) + 4);
  o_t = arma::exp(o_t);
  o_t = arma::ones(3, 1) / (arma::ones(3, 1) + o_t);

  // Expected output for the second input.
  expectedOutput = z_t % expectedOutput + (arma::ones(3, 1) - z_t) % o_t;

  BOOST_REQUIRE_LE(arma::as_scalar(arma::trans(output) * expectedOutput), 1e-2);
}

/**
 * Simple concat module test.
 */
BOOST_AUTO_TEST_CASE(SimpleConcatLayerTest)
{
  arma::mat output, input, delta, error;

  Linear<> moduleA(10, 10);
  moduleA.Parameters().randu();
  moduleA.Reset();

  Linear<> moduleB(10, 10);
  moduleB.Parameters().randu();
  moduleB.Reset();

  Concat<> module;
  module.Add(moduleA);
  module.Add(moduleB);

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_CLOSE(arma::accu(
      moduleA.Parameters().submat(100, 0, moduleA.Parameters().n_elem - 1, 0)) +
      arma::accu(moduleB.Parameters().submat(100, 0,
      moduleB.Parameters().n_elem - 1, 0)),
      arma::accu(output.col(0)), 1e-3);

  // Test the Backward function.
  error = arma::zeros(20, 1);
  module.Backward(std::move(input), std::move(error), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Test to check Concat layer along different axes.
 */
BOOST_AUTO_TEST_CASE(ConcatAlongAxisTest)
{
  arma::mat output, input, error, outputA, outputB;
  size_t inputWidth = 4, inputHeight = 4, inputChannel = 2;
  size_t outputWidth, outputHeight, outputChannel = 2;
  size_t kW = 3, kH = 3;
  size_t batch = 1;

  // Using Convolution<> layer as inout to Concat<> layer.
  // Compute the output shape of convolution layer.
  outputWidth  = (inputWidth - kW) + 1;
  outputHeight = (inputHeight - kH) + 1;

  input = arma::ones(inputWidth * inputHeight * inputChannel, batch);

  Convolution<> moduleA(inputChannel, outputChannel, kW, kH, 1, 1, 0, 0,
      inputWidth, inputHeight);
  Convolution<> moduleB(inputChannel, outputChannel, kW, kH, 1, 1, 0, 0,
      inputWidth, inputHeight);

  moduleA.Reset();
  moduleA.Parameters().randu();
  moduleB.Reset();
  moduleB.Parameters().randu();

  // Compute output of each layer.
  moduleA.Forward(std::move(input), std::move(outputA));
  moduleB.Forward(std::move(input), std::move(outputB));

  arma::cube A(outputA.memptr(), outputWidth, outputHeight, outputChannel);
  arma::cube B(outputB.memptr(), outputWidth, outputHeight, outputChannel);

  error = arma::ones(outputWidth * outputHeight * outputChannel * 2, 1);

  for (size_t axis = 0; axis < 3; ++axis)
  {
    size_t x = 1, y = 1, z = 1;
    arma::cube calculatedOut;
    if (axis == 0)
    {
      calculatedOut.set_size(2 * outputWidth, outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
          arma::mat aMat = A.slice(i);
          arma::mat bMat = B.slice(i);
          calculatedOut.slice(i) = arma::join_cols(aMat, bMat);
      }
      x = 2;
    }
    if (axis == 1)
    {
      calculatedOut.set_size(outputWidth, 2 * outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
          arma::mat aMat = A.slice(i);
          arma::mat bMat = B.slice(i);
          calculatedOut.slice(i) = arma::join_rows(aMat, bMat);
      }
      y = 2;
    }
    if (axis == 2)
    {
      calculatedOut = arma::join_slices(A, B);
      z = 2;
    }

    // Compute output of Concat<> layer.
    arma::Row<size_t> inputSize{outputWidth, outputHeight, outputChannel};
    Concat<> module(inputSize, axis);
    module.Add(moduleA);
    module.Add(moduleB);
    module.Forward(std::move(input), std::move(output));
    arma::cube concatOut(output.memptr(), x * outputWidth,
        y * outputHeight, z * outputChannel);

    // Verify if the output reshaped to cubes are similar.
    CheckMatrices(concatOut, calculatedOut, 1e-12);
  }
}

/**
 * Concat layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientConcatLayerTest)
{
  // Concat function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);

      concat = new Concat<>(true);
      concat->Add<Linear<> >(10, 2);
      model->Add(concat);

      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Concat<>* concat;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple concatenate module test.
 */
BOOST_AUTO_TEST_CASE(SimpleConcatenateLayerTest)
{
  arma::mat input = arma::ones(5, 1);
  arma::mat output, delta;

  Concatenate<> module;
  module.Concat() = arma::ones(5, 1) * 0.5;

  // Test the Forward function.
  module.Forward(std::move(input), std::move(output));

  BOOST_REQUIRE_EQUAL(arma::accu(output), 7.5);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 5);
}

/**
 * Concatenate layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientConcatenateLayerTest)
{
  // Concatenate function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 5);

      arma::mat concat = arma::ones(5, 1);
      concatenate = new Concatenate<>();
      concatenate->Concat() = concat;
      model->Add(concatenate);

      model->Add<Linear<> >(10, 5);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Concatenate<>* concatenate;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple lookup module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLookupLayerTest)
{
  arma::mat output, input, delta, gradient;
  Lookup<> module(10, 5);
  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(2, 1);
  input(0) = 1;
  input(1) = 3;

  module.Forward(std::move(input), std::move(output));

  // The Lookup module uses index - 1 for the cols.
  const double outputSum = arma::accu(module.Parameters().col(0)) +
      arma::accu(module.Parameters().col(2));

  BOOST_REQUIRE_CLOSE(outputSum, arma::accu(output), 1e-3);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(input), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(input));

  // Test the Gradient function.
  arma::mat error = arma::ones(2, 5);
  error = error.t();
  error.col(1) *= 0.5;

  module.Gradient(std::move(input), std::move(error), std::move(gradient));

  // The Lookup module uses index - 1 for the cols.
  const double gradientSum = arma::accu(gradient.col(0)) +
      arma::accu(gradient.col(2));

  BOOST_REQUIRE_CLOSE(gradientSum, arma::accu(error), 1e-3);
  BOOST_REQUIRE_CLOSE(arma::accu(gradient), arma::accu(error), 1e-3);
}

/**
 * Simple LogSoftMax module test.
 */
BOOST_AUTO_TEST_CASE(SimpleLogSoftmaxLayerTest)
{
  arma::mat output, input, error, delta;
  LogSoftMax<> module;

  // Test the Forward function.
  input = arma::mat("0.5; 0.5");
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_SMALL(arma::accu(arma::abs(
    arma::mat("-0.6931; -0.6931") - output)), 1e-3);

  // Test the Backward function.
  error = arma::zeros(input.n_rows, input.n_cols);
  // Assume LogSoftmax layer is always associated with NLL output layer.
  error(1, 0) = -1;
  module.Backward(std::move(input), std::move(error), std::move(delta));
  BOOST_REQUIRE_SMALL(arma::accu(arma::abs(
      arma::mat("1.6487; 0.6487") - delta)), 1e-3);
}

/*
 * Simple test for the BilinearInterpolation layer
 */
BOOST_AUTO_TEST_CASE(SimpleBilinearInterpolationLayerTest)
{
  // Tested output against tensorflow.image.resize_bilinear()
  arma::mat input, output, unzoomedOutput, expectedOutput;
  size_t inRowSize = 2;
  size_t inColSize = 2;
  size_t outRowSize = 5;
  size_t outColSize = 5;
  size_t depth = 1;
  input.zeros(inRowSize * inColSize * depth, 1);
  input[0] = 1.0;
  input[1] = input[2] = 2.0;
  input[3] = 3.0;
  BilinearInterpolation<> layer(inRowSize, inColSize, outRowSize, outColSize,
      depth);
  expectedOutput = arma::mat("1.0000 1.4000 1.8000 2.0000 2.0000 \
      1.4000 1.8000 2.2000 2.4000 2.4000 \
      1.8000 2.2000 2.6000 2.8000 2.8000 \
      2.0000 2.4000 2.8000 3.0000 3.0000 \
      2.0000 2.4000 2.8000 3.0000 3.0000");
  expectedOutput.reshape(25, 1);
  layer.Forward(std::move(input), std::move(output));
  CheckMatrices(output - expectedOutput, arma::zeros(output.n_rows), 1e-12);

  expectedOutput = arma::mat("1.0000 1.9000 1.9000 2.8000");
  expectedOutput.reshape(4, 1);
  layer.Backward(std::move(output), std::move(output),
      std::move(unzoomedOutput));
  CheckMatrices(unzoomedOutput - expectedOutput,
      arma::zeros(input.n_rows), 1e-12);
}

/**
 * Tests the BatchNorm Layer, compares the layers parameters with
 * the values from another implementation.
 * Link to the implementation - http://cthorey.github.io./backpropagation/
 */
BOOST_AUTO_TEST_CASE(BatchNormTest)
{
  arma::mat input, output;
  input << 5.1 << 3.5 << 1.4 << arma::endr
        << 4.9 << 3.0 << 1.4 << arma::endr
        << 4.7 << 3.2 << 1.3 << arma::endr;

  BatchNorm<> model(input.n_rows);
  model.Reset();

  // Non-Deteministic Forward Pass Test.
  model.Deterministic() = false;
  model.Forward(std::move(input), std::move(output));
  arma::mat result;
  result << 1.1658 << 0.1100 << -1.2758 << arma::endr
         << 1.2579 << -0.0699 << -1.1880 << arma::endr
         << 1.1737 << 0.0958 << -1.2695 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Deterministic Forward Pass test.
  output = model.TrainingMean();
  result << 3.33333333 << arma::endr
         << 3.1 << arma::endr
         << 3.06666666 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  output = model.TrainingVariance();
  result << 2.2956 << arma::endr
         << 2.0467 << arma::endr
         << 1.9356 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  model.Deterministic() = true;
  model.Forward(std::move(input), std::move(output));

  result << 1.1658 << 0.1100 << -1.2757 << arma::endr
         << 1.2579 << -0.0699 << -1.1880 << arma::endr
         << 1.1737 << 0.0958 << -1.2695 << arma::endr;

  CheckMatrices(output, result, 1e-1);
}

/**
 * BatchNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientBatchNormTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randn(10, 256);
      arma::mat target;
      target.ones(1, 256);

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<BatchNorm<> >(10);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 256, false);
      model->Gradient(model->Parameters(), 0, gradient, 256);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * VirtualBatchNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientVirtualBatchNormTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randn(5, 256);
      arma::mat referenceBatch = arma::mat(input.memptr(), input.n_rows, 16);
      arma::mat target;
      target.ones(1, 256);

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(5, 5);
      model->Add<VirtualBatchNorm<> >(referenceBatch, 5);
      model->Add<Linear<> >(5, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 256, false);
      model->Gradient(model->Parameters(), 0, gradient, 256);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * MiniBatchDiscrimination layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(MiniBatchDiscriminationTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randn(5, 4);
      arma::mat target;
      target.ones(1, 4);

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(5, 5);
      model->Add<MiniBatchDiscrimination<> >(5, 10, 16);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      return model->EvaluateWithGradient(model->Parameters(), 0, gradient, 4);
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple Transposed Convolution layer test.
 */
BOOST_AUTO_TEST_CASE(SimpleTransposedConvolutionLayerTest)
{
  arma::mat output, input, delta;

  TransposedConvolution<> module1(1, 1, 3, 3, 1, 1, 0, 0, 4, 4, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 15, 16);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Parameters()(0) = 1.0;
  module1.Parameters()(8) = 2.0;
  module1.Reset();
  module1.Forward(std::move(input), std::move(output));
  // Value calculated using tensorflow.nn.conv2d_transpose()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 360.0);

  // Test the backward function.
  module1.Backward(std::move(input), std::move(output), std::move(delta));
  // Value calculated using tensorflow.nn.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 720.0);

  TransposedConvolution<> module2(1, 1, 4, 4, 1, 1, 1, 1, 5, 5, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module2.Parameters() = arma::mat(16 + 1, 1, arma::fill::zeros);
  module2.Parameters()(0) = 1.0;
  module2.Parameters()(3) = 1.0;
  module2.Parameters()(6) = 1.0;
  module2.Parameters()(9) = 1.0;
  module2.Parameters()(12) = 1.0;
  module2.Parameters()(15) = 2.0;
  module2.Reset();
  module2.Forward(std::move(input), std::move(output));
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 1512.0);

  // Test the backward function.
  module2.Backward(std::move(input), std::move(output), std::move(delta));
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 6504.0);

  TransposedConvolution<> module3(1, 1, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module3.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module3.Parameters()(1) = 2.0;
  module3.Parameters()(2) = 4.0;
  module3.Parameters()(3) = 3.0;
  module3.Parameters()(8) = 1.0;
  module3.Reset();
  module3.Forward(std::move(input), std::move(output));
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 2370.0);

  // Test the backward function.
  module3.Backward(std::move(input), std::move(output), std::move(delta));
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 19154.0);

  TransposedConvolution<> module4(1, 1, 3, 3, 1, 1, 0, 0, 5, 5, 7, 7);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module4.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module4.Parameters()(2) = 2.0;
  module4.Parameters()(4) = 4.0;
  module4.Parameters()(6) = 6.0;
  module4.Parameters()(8) = 8.0;
  module4.Reset();
  module4.Forward(std::move(input), std::move(output));
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 6000.0);

  // Test the backward function.
  module4.Backward(std::move(input), std::move(output), std::move(delta));
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 86208.0);

  TransposedConvolution<> module5(1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module5.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module5.Parameters()(2) = 8.0;
  module5.Parameters()(4) = 6.0;
  module5.Parameters()(6) = 4.0;
  module5.Parameters()(8) = 2.0;
  module5.Reset();
  module5.Forward(std::move(input), std::move(output));
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 120.0);

  // Test the backward function.
  module5.Backward(std::move(input), std::move(output), std::move(delta));
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 960.0);

  TransposedConvolution<> module6(1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module6.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module6.Parameters()(0) = 8.0;
  module6.Parameters()(3) = 6.0;
  module6.Parameters()(6) = 2.0;
  module6.Parameters()(8) = 4.0;
  module6.Reset();
  module6.Forward(std::move(input), std::move(output));
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 410.0);

  // Test the backward function.
  module6.Backward(std::move(input), std::move(output), std::move(delta));
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 4444.0);

  TransposedConvolution<> module7(1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module7.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module7.Parameters()(0) = 8.0;
  module7.Parameters()(2) = 6.0;
  module7.Parameters()(4) = 2.0;
  module7.Parameters()(8) = 4.0;
  module7.Reset();
  module7.Forward(std::move(input), std::move(output));
  // Value calculated using torch.nn.functional.conv_transpose2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 606.0);

  module7.Backward(std::move(input), std::move(output), std::move(delta));
  // Value calculated using torch.nn.functional.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 7732.0);
}

/**
 * Transposed Convolution layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientTransposedConvolutionLayerTest)
{
  // Add function gradient instantiation.
  // To make this test robust, check it five times.
  bool pass = false;
  for (size_t trial = 0; trial < 5; trial++)
  {
    struct GradientFunction
    {
      GradientFunction()
      {
        input = arma::linspace<arma::colvec>(0, 35, 36);
        target = arma::mat("1");

        model = new FFN<NegativeLogLikelihood<>, RandomInitialization>();
        model->Predictors() = input;
        model->Responses() = target;
        model->Add<TransposedConvolution<> >
            (1, 1, 3, 3, 2, 2, 1, 1, 6, 6, 12, 12);
        model->Add<LogSoftMax<> >();
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

      FFN<NegativeLogLikelihood<>, RandomInitialization>* model;
      arma::mat input, target;
    } function;

    if (CheckGradient(function) < 1e-3)
    {
      pass = true;
      break;
    }
  }
  BOOST_REQUIRE_EQUAL(pass, true);
}

/**
 * Simple MultiplyMerge module test.
 */
BOOST_AUTO_TEST_CASE(SimpleMultiplyMergeLayerTest)
{
  arma::mat output, input, delta;
  input = arma::ones(10, 1);

  for (size_t i = 0; i < 5; ++i)
  {
    MultiplyMerge<> module(false, false);
    const size_t numMergeModules = math::RandInt(2, 10);
    for (size_t m = 0; m < numMergeModules; ++m)
    {
      IdentityLayer<> identityLayer;
      identityLayer.Forward(std::move(input),
          std::move(identityLayer.OutputParameter()));

      module.Add<IdentityLayer<> >(identityLayer);
    }

    // Test the Forward function.
    module.Forward(std::move(input), std::move(output));
    BOOST_REQUIRE_EQUAL(10, arma::accu(output));

    // Test the Backward function.
    module.Backward(std::move(input), std::move(output), std::move(delta));
    BOOST_REQUIRE_EQUAL(arma::accu(output), arma::accu(delta));
  }
}

/**
 * Simple Atrous Convolution layer test.
 */
BOOST_AUTO_TEST_CASE(SimpleAtrousConvolutionLayerTest)
{
  arma::mat output, input, delta;

  AtrousConvolution<> module1(1, 1, 3, 3, 1, 1, 0, 0, 7, 7, 2, 2);
  // Test the Forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Parameters()(0) = 1.0;
  module1.Parameters()(8) = 2.0;
  module1.Reset();
  module1.Forward(std::move(input), std::move(output));
  // Value calculated using tensorflow.nn.atrous_conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 792.0);

  // Test the Backward function.
  module1.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 2376);

  AtrousConvolution<> module2(1, 1, 3, 3, 2, 2, 0, 0, 7, 7, 2, 2);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module2.Parameters()(0) = 1.0;
  module2.Parameters()(3) = 1.0;
  module2.Parameters()(6) = 1.0;
  module2.Reset();
  module2.Forward(std::move(input), std::move(output));
  // Value calculated using tensorflow.nn.conv2d()
  BOOST_REQUIRE_EQUAL(arma::accu(output), 264.0);

  // Test the backward function.
  module2.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 792.0);
}

/**
 * Atrous Convolution layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientAtrousConvolutionLayerTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::linspace<arma::colvec>(0, 35, 36);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, RandomInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<AtrousConvolution<> >(1, 1, 3, 3, 1, 1, 0, 0, 6, 6, 2, 2);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  // TODO: this tolerance seems far higher than necessary.  The implementation
  // should be checked.
  BOOST_REQUIRE_LE(CheckGradient(function), 0.2);
}

/**
 * Test the functions to access and modify the parameters of the
 * AtrousConvolution layer.
 */
BOOST_AUTO_TEST_CASE(AtrousConvolutionLayerParametersTest)
{
  // Parameter order for the constructor: inSize, outSize, kW, kH, dW, dH, padW,
  // padH, inputWidth, inputHeight, dilationW, dilationH, paddingType ("none").
  AtrousConvolution<> layer1(1, 2, 3, 4, 5, 6, std::make_tuple(7, 8),
      std::make_tuple(9, 10), 11, 12, 13, 14);
  AtrousConvolution<> layer2(2, 3, 4, 5, 6, 7, std::make_tuple(8, 9),
      std::make_tuple(10, 11), 12, 13, 14, 15);

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), 11);
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), 12);
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), 3);
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), 4);
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), 5);
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), 6);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHTop(), 9);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHBottom(), 10);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWLeft(), 7);
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWRight(), 8);
  BOOST_REQUIRE_EQUAL(layer1.DilationWidth(), 13);
  BOOST_REQUIRE_EQUAL(layer1.DilationHeight(), 14);

  // Now modify the parameters to match the second layer.
  layer1.InputWidth() = 12;
  layer1.InputHeight() = 13;
  layer1.KernelWidth() = 4;
  layer1.KernelHeight() = 5;
  layer1.StrideWidth() = 6;
  layer1.StrideHeight() = 7;
  layer1.Padding().PadHTop() = 10;
  layer1.Padding().PadHBottom() = 11;
  layer1.Padding().PadWLeft() = 8;
  layer1.Padding().PadWRight() = 9;
  layer1.DilationWidth() = 14;
  layer1.DilationHeight() = 15;

  // Now ensure all results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), layer2.InputWidth());
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), layer2.InputHeight());
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), layer2.KernelWidth());
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), layer2.KernelHeight());
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), layer2.StrideWidth());
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), layer2.StrideHeight());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHTop(), layer2.Padding().PadHTop());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadHBottom(),
                      layer2.Padding().PadHBottom());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWLeft(),
                      layer2.Padding().PadWLeft());
  BOOST_REQUIRE_EQUAL(layer1.Padding().PadWRight(),
                      layer2.Padding().PadWRight());
  BOOST_REQUIRE_EQUAL(layer1.DilationWidth(), layer2.DilationWidth());
  BOOST_REQUIRE_EQUAL(layer1.DilationHeight(), layer2.DilationHeight());
}

/**
 * Test that the padding options are working correctly in Atrous Convolution
 * layer.
 */
BOOST_AUTO_TEST_CASE(AtrousConvolutionLayerPaddingTest)
{
  arma::mat output, input, delta;

  // Check valid padding option.
  AtrousConvolution<> module1(1, 1, 3, 3, 1, 1,
      std::tuple<size_t, size_t>(1, 1), std::tuple<size_t, size_t>(1, 1), 7, 7,
      2, 2, "valid");

  // Test the Forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Reset();
  module1.Forward(std::move(input), std::move(output));

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 9);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the Backward function.
  module1.Backward(std::move(input), std::move(output), std::move(delta));

  // Check same padding option.
  AtrousConvolution<> module2(1, 1, 3, 3, 1, 1,
      std::tuple<size_t, size_t>(0, 0), std::tuple<size_t, size_t>(0, 0), 7, 7,
      2, 2, "same");

  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module2.Reset();
  module2.Forward(std::move(input), std::move(output));

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 49);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the backward function.
  module2.Backward(std::move(input), std::move(output), std::move(delta));
}

/**
 * Tests the LayerNorm layer.
 */
BOOST_AUTO_TEST_CASE(LayerNormTest)
{
  arma::mat input, output;
  input << 5.1 << 3.5 << arma::endr
        << 4.9 << 3.0 << arma::endr
        << 4.7 << 3.2 << arma::endr;

  LayerNorm<> model(input.n_rows);
  model.Reset();

  model.Forward(std::move(input), std::move(output));
  arma::mat result;
  result << 1.2247 << 1.2978 << arma::endr
         << 0 << -1.1355 << arma::endr
         << -1.2247 << -0.1622 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  output = model.Mean();
  result << 4.9000 << 3.2333 << arma::endr;

  CheckMatrices(output, result, 1e-1);
  result.clear();

  output = model.Variance();
  result << 0.0267 << 0.0422 << arma::endr;

  CheckMatrices(output, result, 1e-1);
}

/**
 * LayerNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLayerNormTest)
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randn(10, 256);
      arma::mat target;
      target.ones(1, 256);

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      model->Add<LayerNorm<> >(10);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 256, false);
      model->Gradient(model->Parameters(), 0, gradient, 256);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test if the AddMerge layer is able to forward the
 * Forward/Backward/Gradient calls.
 */
BOOST_AUTO_TEST_CASE(AddMergeRunTest)
{
  arma::mat output, input, delta, error;

  AddMerge<> module(true, true);

  Linear<>* linear = new Linear<>(10, 10);
  module.Add(linear);

  linear->Parameters().randu();
  linear->Reset();

  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));

  double parameterSum = arma::accu(linear->Parameters().submat(
      100, 0, linear->Parameters().n_elem - 1, 0));

  // Test the Backward function.
  module.Backward(std::move(input), std::move(input), std::move(delta));

  // Clean up before we break,
  delete linear;

  BOOST_REQUIRE_CLOSE(parameterSum, arma::accu(output), 1e-3);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Test if the MultiplyMerge layer is able to forward the
 * Forward/Backward/Gradient calls.
 */
BOOST_AUTO_TEST_CASE(MultiplyMergeRunTest)
{
  arma::mat output, input, delta, error;

  MultiplyMerge<> module(true, true);

  Linear<>* linear = new Linear<>(10, 10);
  module.Add(linear);

  linear->Parameters().randu();
  linear->Reset();

  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));

  double parameterSum = arma::accu(linear->Parameters().submat(
      100, 0, linear->Parameters().n_elem - 1, 0));

  // Test the Backward function.
  module.Backward(std::move(input), std::move(input), std::move(delta));

  // Clean up before we break,
  delete linear;

  BOOST_REQUIRE_CLOSE(parameterSum, arma::accu(output), 1e-3);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Simple subview module test.
 */
BOOST_AUTO_TEST_CASE(SimpleSubviewLayerTest)
{
  arma::mat output, input, delta, outputMat;
  Subview<> moduleRow(1, 10, 19);

  // Test the Forward function for a vector.
  input = arma::ones(20, 1);
  moduleRow.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(output.n_rows, 10);

  Subview<> moduleMat(4, 3, 6, 0, 2);

  // Test the Forward function for a matrix.
  input = arma::ones(20, 8);
  moduleMat.Forward(std::move(input), std::move(outputMat));
  BOOST_REQUIRE_EQUAL(outputMat.n_rows, 12);
  BOOST_REQUIRE_EQUAL(outputMat.n_cols, 2);

  // Test the Backward function.
  moduleMat.Backward(std::move(input), std::move(input), std::move(delta));
  BOOST_REQUIRE_EQUAL(accu(delta), 160);
  BOOST_REQUIRE_EQUAL(delta.n_rows, 20);
}

/**
 * Subview index test.
 */
BOOST_AUTO_TEST_CASE(SubviewIndexTest)
{
  arma::mat outputEnd, outputMid, outputStart, input, delta;
  input = arma::linspace<arma::vec>(1, 20, 20);

  // Slicing from the initial indices.
  Subview<> moduleStart(1, 0, 9);
  arma::mat subStart = arma::linspace<arma::vec>(1, 10, 10);

  moduleStart.Forward(std::move(input), std::move(outputStart));
  CheckMatrices(outputStart, subStart);

  // Slicing from the mid indices.
  Subview<> moduleMid(1, 6, 15);
  arma::mat subMid = arma::linspace<arma::vec>(7, 16, 10);

  moduleMid.Forward(std::move(input), std::move(outputMid));
  CheckMatrices(outputMid, subMid);

  // Slicing from the end indices.
  Subview<> moduleEnd(1, 10, 19);
  arma::mat subEnd = arma::linspace<arma::vec>(11, 20, 10);

  moduleEnd.Forward(std::move(input), std::move(outputEnd));
  CheckMatrices(outputEnd, subEnd);
}

/**
 * Subview batch test.
 */
BOOST_AUTO_TEST_CASE(SubviewBatchTest)
{
  arma::mat output, input, outputCol, outputMat, outputDef;

  // All rows selected.
  Subview<> moduleCol(1, 0, 19);

  // Test with inSize 1.
  input = arma::ones(20, 8);
  moduleCol.Forward(std::move(input), std::move(outputCol));
  CheckMatrices(outputCol, input);

  // Few rows and columns selected.
  Subview<> moduleMat(4, 3, 6, 0, 2);

  // Test with inSize greater than 1.
  moduleMat.Forward(std::move(input), std::move(outputMat));
  output = arma::ones(12, 2);
  CheckMatrices(outputMat, output);

  // endCol changed to 3 by default.
  Subview<> moduleDef(4, 1, 6, 0, 4);

  // Test with inSize greater than 1 and endCol >= inSize.
  moduleDef.Forward(std::move(input), std::move(outputDef));
  output = arma::ones(24, 2);
  CheckMatrices(outputDef, output);
}

/*
 * Simple Reparametrization module test.
 */
BOOST_AUTO_TEST_CASE(SimpleReparametrizationLayerTest)
{
  arma::mat input, output, delta;
  Reparametrization<> module(5);

  // Test the Forward function. As the mean is zero and the standard
  // deviation is small, after multiplying the gaussian sample, the
  // output should be small enough.
  input = join_cols(arma::ones<arma::mat>(5, 1) * -15,
      arma::zeros<arma::mat>(5, 1));
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_LE(arma::accu(output), 1e-5);

  // Test the Backward function.
  arma::mat gy = arma::zeros<arma::mat>(5, 1);
  module.Backward(std::move(input), std::move(gy), std::move(delta));
  BOOST_REQUIRE(arma::accu(delta) != 0); // klBackward will be added.
}

/**
 * Reparametrization module stochastic boolean test.
 */
BOOST_AUTO_TEST_CASE(ReparametrizationLayerStochasticTest)
{
  arma::mat input, outputA, outputB;
  Reparametrization<> module(5, false);

  input = join_cols(arma::ones<arma::mat>(5, 1),
      arma::zeros<arma::mat>(5, 1));

  // Test if two forward passes generate same output.
  module.Forward(std::move(input), std::move(outputA));
  module.Forward(std::move(input), std::move(outputB));

  CheckMatrices(outputA, outputB);
}

/**
 * Reparametrization module includeKl boolean test.
 */
BOOST_AUTO_TEST_CASE(ReparametrizationLayerIncludeKlTest)
{
  arma::mat input, output, gy, delta;
  Reparametrization<> module(5, true, false);

  input = join_cols(arma::ones<arma::mat>(5, 1),
      arma::zeros<arma::mat>(5, 1));
  module.Forward(std::move(input), std::move(output));

  // As KL divergence is not included, with the above inputs, the delta
  // matrix should be all zeros.
  gy = arma::zeros(output.n_rows, output.n_cols);
  module.Backward(std::move(output), std::move(gy), std::move(delta));

  BOOST_REQUIRE_EQUAL(arma::accu(std::move(delta)), 0);
}

/**
 * Jacobian Reparametrization module test.
 */
BOOST_AUTO_TEST_CASE(JacobianReparametrizationLayerTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t inputElementsHalf = math::RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElementsHalf * 2, 1);

    Reparametrization<> module(inputElementsHalf, false, false);

    double error = JacobianTest(module, input);
    BOOST_REQUIRE_LE(error, 1e-5);
  }
}

/**
 * Reparametrization layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientReparametrizationLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 6);
      model->Add<Reparametrization<> >(3, false, true, 1);
      model->Add<Linear<> >(3, 2);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Reparametrization layer beta numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientReparametrizationLayerBetaTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 2);
      target = arma::mat("1 1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 6);
      // Use a value of beta not equal to 1.
      model->Add<Reparametrization<> >(3, false, true, 2);
      model->Add<Linear<> >(3, 2);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Simple residual module test.
 */
BOOST_AUTO_TEST_CASE(SimpleResidualLayerTest)
{
  arma::mat outputA, outputB, input, deltaA, deltaB;

  Sequential<>* sequential = new Sequential<>(true);
  Residual<>* residual = new Residual<>(true);

  Linear<>* linearA = new Linear<>(10, 10);
  linearA->Parameters().randu();
  linearA->Reset();
  Linear<>* linearB = new Linear<>(10, 10);
  linearB->Parameters().randu();
  linearB->Reset();

  // Add the same layers (with the same parameters) to both Sequential and
  // Residual object.
  sequential->Add(linearA);
  sequential->Add(linearB);

  residual->Add(linearA);
  residual->Add(linearB);

  // Test the Forward function (pass the same input to both).
  input = arma::randu(10, 1);
  sequential->Forward(std::move(input), std::move(outputA));
  residual->Forward(std::move(input), std::move(outputB));

  CheckMatrices(outputA, outputB - input);

  // Test the Backward function (pass the same error to both).
  sequential->Backward(std::move(input), std::move(input), std::move(deltaA));
  residual->Backward(std::move(input), std::move(input), std::move(deltaB));

  CheckMatrices(deltaA, deltaB - input);

  delete sequential;
  delete residual;
  delete linearA;
  delete linearB;
}

/**
 * Simple Highway module test.
 */
BOOST_AUTO_TEST_CASE(SimpleHighwayLayerTest)
{
  arma::mat outputA, outputB, input, deltaA, deltaB;
  Sequential<>* sequential = new Sequential<>(true);
  Highway<>* highway = new Highway<>(10, true);
  highway->Parameters().zeros();
  highway->Reset();

  Linear<>* linearA = new Linear<>(10, 10);
  linearA->Parameters().randu();
  linearA->Reset();
  Linear<>* linearB = new Linear<>(10, 10);
  linearB->Parameters().randu();
  linearB->Reset();

  // Add the same layers (with the same parameters) to both Sequential and
  // Highway object.
  highway->Add(linearA);
  highway->Add(linearB);
  sequential->Add(linearA);
  sequential->Add(linearB);

  // Test the Forward function (pass the same input to both).
  input = arma::randu(10, 1);
  sequential->Forward(std::move(input), std::move(outputA));
  highway->Forward(std::move(input), std::move(outputB));

  CheckMatrices(outputB, input * 0.5 + outputA * 0.5);

  delete sequential;
  delete highway;
  delete linearA;
  delete linearB;
}

/**
 * Sequential layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientHighwayLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(5, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(5, 10);

      highway = new Highway<>(10);
      highway->Add<Linear<> >(10, 10);
      highway->Add<ReLULayer<> >();
      highway->Add<Linear<> >(10, 10);
      highway->Add<ReLULayer<> >();

      model->Add(highway);
      model->Add<Linear<> >(10, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      highway->DeleteModules();
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Highway<>* highway;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Sequential layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientSequentialLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 10);
      sequential = new Sequential<>();
      sequential->Add<Linear<> >(10, 10);
      sequential->Add<ReLULayer<> >();
      sequential->Add<Linear<> >(10, 5);
      sequential->Add<ReLULayer<> >();

      model->Add(sequential);
      model->Add<Linear<> >(5, 2);
      model->Add<LogSoftMax<> >();
    }

    ~GradientFunction()
    {
      sequential->DeleteModules();
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    Sequential<>* sequential;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * WeightNorm layer numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientWeightNormLayerTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::mat("1");

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<Linear<> >(10, 10);

      Linear<>* linear = new Linear<>(10, 2);
      weightNorm = new WeightNorm<>(linear);

      model->Add(weightNorm);
      model->Add<LogSoftMax<> >();
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

    FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>* model;
    WeightNorm<>* weightNorm;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/**
 * Test if the WeightNorm layer is able to forward the
 * Forward/Backward/Gradient calls.
 */
BOOST_AUTO_TEST_CASE(WeightNormRunTest)
{
  arma::mat output, input, delta, error;

  Linear<>* linear = new Linear<>(10, 10);

  WeightNorm<> module(linear);

  module.Parameters().randu();
  module.Reset();

  linear->Bias().zeros();

  input = arma::zeros(10, 1);
  module.Forward(std::move(input), std::move(output));

  // Test the Backward function.
  module.Backward(std::move(input), std::move(input), std::move(delta));

  BOOST_REQUIRE_EQUAL(0, arma::accu(output));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

// General ANN serialization test.
template<typename LayerType>
void ANNLayerSerializationTest(LayerType& layer)
{
  arma::mat input(5, 100, arma::fill::randu);
  arma::mat output(5, 100, arma::fill::randu);

  FFN<NegativeLogLikelihood<>, ann::RandomInitialization> model;
  model.Add<Linear<>>(input.n_rows, 10);
  model.Add<LayerType>(layer);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(10, output.n_rows);
  model.Add<LogSoftMax<>>();

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  model.Train(input, output, opt);

  arma::mat originalOutput;
  model.Predict(input, originalOutput);

  // Now serialize the model.
  FFN<NegativeLogLikelihood<>, ann::RandomInitialization> xmlModel, textModel,
      binaryModel;
  SerializeObjectAll(model, xmlModel, textModel, binaryModel);

  // Ensure that predictions are the same.
  arma::mat modelOutput, xmlOutput, textOutput, binaryOutput;
  model.Predict(input, modelOutput);
  xmlModel.Predict(input, xmlOutput);
  textModel.Predict(input, textOutput);
  binaryModel.Predict(input, binaryOutput);

  CheckMatrices(originalOutput, modelOutput, 1e-5);
  CheckMatrices(originalOutput, xmlOutput, 1e-5);
  CheckMatrices(originalOutput, textOutput, 1e-5);
  CheckMatrices(originalOutput, binaryOutput, 1e-5);
}

/**
 * Simple serialization test for batch normalization layer.
 */
BOOST_AUTO_TEST_CASE(BatchNormSerializationTest)
{
  BatchNorm<> layer(10);
  ANNLayerSerializationTest(layer);
}

/**
 * Simple serialization test for layer normalization layer.
 */
BOOST_AUTO_TEST_CASE(LayerNormSerializationTest)
{
  LayerNorm<> layer(10);
  ANNLayerSerializationTest(layer);
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Convolution layer work.
 */
BOOST_AUTO_TEST_CASE(ConvolutionLayerParametersTest)
{
  // Parameter order: inSize, outSize, kW, kH, dW, dH, padW, padH, inputWidth,
  // inputHeight, paddingType.
  Convolution<> layer1(1, 2, 3, 4, 5, 6, std::tuple<size_t, size_t>(7, 8),
      std::tuple<size_t, size_t>(9, 10), 11, 12, "none");
  Convolution<> layer2(2, 3, 4, 5, 6, 7, std::tuple<size_t, size_t>(8, 9),
      std::tuple<size_t, size_t>(10, 11), 12, 13, "none");

  // Make sure we can get the parameters successfully.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), 11);
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), 12);
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), 3);
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), 4);
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), 5);
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), 6);
  BOOST_REQUIRE_EQUAL(layer1.PadWLeft(), 7);
  BOOST_REQUIRE_EQUAL(layer1.PadWRight(), 8);
  BOOST_REQUIRE_EQUAL(layer1.PadHTop(), 9);
  BOOST_REQUIRE_EQUAL(layer1.PadHBottom(), 10);

  // Now modify the parameters to match the second layer.
  layer1.InputWidth() = 12;
  layer1.InputHeight() = 13;
  layer1.KernelWidth() = 4;
  layer1.KernelHeight() = 5;
  layer1.StrideWidth() = 6;
  layer1.StrideHeight() = 7;
  layer1.PadWLeft() = 8;
  layer1.PadWRight() = 9;
  layer1.PadHTop() = 10;
  layer1.PadHBottom() = 11;

  // Now ensure all results are the same.
  BOOST_REQUIRE_EQUAL(layer1.InputWidth(), layer2.InputWidth());
  BOOST_REQUIRE_EQUAL(layer1.InputHeight(), layer2.InputHeight());
  BOOST_REQUIRE_EQUAL(layer1.KernelWidth(), layer2.KernelWidth());
  BOOST_REQUIRE_EQUAL(layer1.KernelHeight(), layer2.KernelHeight());
  BOOST_REQUIRE_EQUAL(layer1.StrideWidth(), layer2.StrideWidth());
  BOOST_REQUIRE_EQUAL(layer1.StrideHeight(), layer2.StrideHeight());
  BOOST_REQUIRE_EQUAL(layer1.PadWLeft(), layer2.PadWLeft());
  BOOST_REQUIRE_EQUAL(layer1.PadWRight(), layer2.PadWRight());
  BOOST_REQUIRE_EQUAL(layer1.PadHTop(), layer2.PadHTop());
  BOOST_REQUIRE_EQUAL(layer1.PadHBottom(), layer2.PadHBottom());
}

/**
 * Test that the padding options are working correctly in Convolution layer.
 */
BOOST_AUTO_TEST_CASE(ConvolutionLayerPaddingTest)
{
  arma::mat output, input, delta;

  // Check valid padding option.
  Convolution<> module1(1, 1, 3, 3, 1, 1, std::tuple<size_t, size_t>(1, 1),
      std::tuple<size_t, size_t>(1, 1), 7, 7, "valid");

  // Test the Forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Reset();
  module1.Forward(std::move(input), std::move(output));

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 25);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the Backward function.
  module1.Backward(std::move(input), std::move(output), std::move(delta));

  // Check same padding option.
  Convolution<> module2(1, 1, 3, 3, 1, 1, std::tuple<size_t, size_t>(0, 0),
      std::tuple<size_t, size_t>(0, 0), 7, 7, "same");

  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 48, 49);
  module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module2.Reset();
  module2.Forward(std::move(input), std::move(output));

  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, 49);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);

  // Test the backward function.
  module2.Backward(std::move(input), std::move(output), std::move(delta));
}

/**
 * Test that the padding options in Transposed Convolution layer.
 */
BOOST_AUTO_TEST_CASE(TransposedConvolutionLayerPaddingTest)
{
  arma::mat output, input, delta;

  TransposedConvolution<> module1(1, 1, 3, 3, 1, 1, 0, 0, 4, 4, 6, 6, "VALID");
  // Test the forward function.
  // Valid Should give the same result.
  input = arma::linspace<arma::colvec>(0, 15, 16);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Reset();
  module1.Forward(std::move(input), std::move(output));
  // Value calculated using tensorflow.nn.conv2d_transpose().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0.0);

  // Test the Backward Function.
  module1.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  // Test Valid for non zero padding.
  TransposedConvolution<> module2(1, 1, 3, 3, 2, 2,
      std::tuple<size_t, size_t>(0, 0), std::tuple<size_t, size_t>(0, 0),
      2, 2, 5, 5, "VALID");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module2.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module2.Parameters()(2) = 8.0;
  module2.Parameters()(4) = 6.0;
  module2.Parameters()(6) = 4.0;
  module2.Parameters()(8) = 2.0;
  module2.Reset();
  module2.Forward(std::move(input), std::move(output));
  // Value calculated using torch.nn.functional.conv_transpose2d().
  BOOST_REQUIRE_EQUAL(arma::accu(output), 120.0);

  // Test the Backward Function.
  module2.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 960.0);

  // Test for same padding type.
  TransposedConvolution<> module3(1, 1, 3, 3, 2, 2, 0, 0, 3, 3, 3, 3, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module3.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module3.Reset();
  module3.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module3.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  // Output shape should equal input.
  TransposedConvolution<> module4(1, 1, 3, 3, 1, 1,
    std::tuple<size_t, size_t>(2, 2), std::tuple<size_t, size_t>(2, 2),
    5, 5, 5, 5, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module4.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module4.Reset();
  module4.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module4.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  TransposedConvolution<> module5(1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module5.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module5.Reset();
  module5.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module5.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);

  TransposedConvolution<> module6(1, 1, 4, 4, 1, 1, 1, 1, 5, 5, 5, 5, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module6.Parameters() = arma::mat(16 + 1, 1, arma::fill::zeros);
  module6.Reset();
  module6.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(output), 0);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the Backward Function.
  module6.Backward(std::move(input), std::move(output), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);
}
BOOST_AUTO_TEST_SUITE_END();
