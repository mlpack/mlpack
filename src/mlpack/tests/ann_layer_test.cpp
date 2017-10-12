/**
 * @file ann_layer_test.cpp
 * @author Marcus Edel
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

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNLayerTest);

// Helper function which calls the Reset function of the given module.
template<class T>
void ResetFunction(
    T& layer,
    typename std::enable_if<HasResetCheck<T, void(T::*)()>::value>::type* = 0)
{
  layer.Reset();
}

template<class T>
void ResetFunction(
    T& /* layer */,
    typename std::enable_if<!HasResetCheck<T, void(T::*)()>::value>::type* = 0)
{
  /* Nothing to do here */
}

// Approximate Jacobian and supposedly-true Jacobian, then compare them
// similarly to before.
template<typename ModuleType>
double JacobianTest(ModuleType& module,
                  arma::mat& input,
                  const double minValue = -2,
                  const double maxValue = -1,
                  const double perturbation = 1e-6)
{
  arma::mat output, outputA, outputB, jacobianA, jacobianB;

  // Initialize the input matrix.
  RandomInitialization init(minValue, maxValue);
  init.Initialize(input, input.n_rows, input.n_cols);

  // Initialize the module parameters.
  ResetFunction(module);

  // Initialize the jacobian matrix.
  module.Forward(std::move(input), std::move(output));
  jacobianA = arma::zeros(input.n_elem, output.n_elem);

  // Share the input paramter matrix.
  arma::mat sin = arma::mat(input.memptr(), input.n_rows, input.n_cols,
      false, false);

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    double original = sin(i);
    sin(i) = original - perturbation;
    module.Forward(std::move(input), std::move(outputA));
    sin(i) = original + perturbation;
    module.Forward(std::move(input), std::move(outputB));
    sin(i) = original;

    outputB -= outputA;
    outputB /= 2 * perturbation;
    jacobianA.row(i) = outputB.t();
  }

  // Initialize the derivative parameter.
  arma::mat deriv = arma::zeros(output.n_rows, output.n_cols);

  // Share the derivative parameter.
  arma::mat derivTemp = arma::mat(deriv.memptr(), deriv.n_rows, deriv.n_cols,
      false, false);

  // Initialize the jacobian matrix.
  jacobianB = arma::zeros(input.n_elem, output.n_elem);

  for (size_t i = 0; i < derivTemp.n_elem; ++i)
  {
    deriv.zeros();
    derivTemp(i) = 1;

    arma::mat delta;
    module.Backward(std::move(input), std::move(deriv), std::move(delta));

    jacobianB.col(i) = delta;
  }

  return arma::max(arma::max(arma::abs(jacobianA - jacobianB)));
}

// Approximate Jacobian and supposedly-true Jacobian, then compare them
// similarly to before.
template<typename ModuleType>
double JacobianPerformanceTest(ModuleType& module,
                               arma::mat& input,
                               arma::mat& target,
                               const double eps = 1e-6)
{
  module.Forward(std::move(input), std::move(target));

  arma::mat delta;
  module.Backward(std::move(input), std::move(target), std::move(delta));

  arma::mat centralDifference = arma::zeros(delta.n_rows, delta.n_cols);
  arma::mat inputTemp = arma::mat(input.memptr(), input.n_rows, input.n_cols,
      false, false);

  arma::mat centralDifferenceTemp = arma::mat(centralDifference.memptr(),
      centralDifference.n_rows, centralDifference.n_cols, false, false);

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    inputTemp(i) = inputTemp(i) + eps;
    double outputA = module.Forward(std::move(input), std::move(target));
    inputTemp(i) = inputTemp(i) - (2 * eps);
    double outputB = module.Forward(std::move(input), std::move(target));

    centralDifferenceTemp(i) = (outputA - outputB) / (2 * eps);
    inputTemp(i) = inputTemp(i) + eps;
  }

  return arma::max(arma::max(arma::abs(centralDifference - delta)));
}

// Simple numerical gradient checker.
template<class FunctionType>
double CheckGradient(FunctionType& function, const double eps = 1e-7)
{
  // Get gradients for the current parameters.
  arma::mat orgGradient, gradient, estGradient;
  function.Gradient(orgGradient);

  estGradient = arma::zeros(orgGradient.n_rows, orgGradient.n_cols);

  // compute numeric approximations to gradient.
  for (size_t i = 0; i < orgGradient.n_elem; ++i)
  {
    double tmp = function.Parameters()(i);

    // Perturb parameter with a positive constant and get costs.
    function.Parameters()(i) += eps;
    double costPlus = function.Gradient(gradient);

    // Perturb parameter with a negative constant and get costs.
    function.Parameters()(i) -= (2 * eps);
    double costMinus = function.Gradient(gradient);

    // Restore the parameter value.
    function.Parameters()(i) = tmp;

    // Compute numerical gradients using the costs calculated above.
    estGradient(i) = (costPlus - costMinus) / (2 * eps);
  }

  // Estimate error of gradient.
  return arma::norm(orgGradient - estGradient) /
      arma::norm(orgGradient + estGradient);
}

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
 * Add layer numerically gradient test.
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

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>(
          input, target);
      model->Add<IdentityLayer<> >();
      model->Add<Add<> >(10);
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
  // Initialize the probability of setting a value to zero and the scale
  // parameter.
  const double p = 0.2;
  const double scale = 1.0 / (1.0 - p);

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
  module.Rescale() = false;
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_EQUAL(arma::accu(input), arma::accu(output));

  // Test the Forward function.
  module.Rescale() = true;
  module.Forward(std::move(input), std::move(output));
  BOOST_REQUIRE_CLOSE(arma::accu(input) * scale, arma::accu(output), 1e-3);
}

/**
 * Perform dropout x times using ones as input, sum the number of ones and
 * validate that the layer is is producing approximately the right number of
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
 * Linear layer numerically gradient test.
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

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>(
          input, target);
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 2);
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
 * LinearNoBias layer numerically gradient test.
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

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>(
          input, target);
      model->Add<IdentityLayer<> >();
      model->Add<LinearNoBias<> >(10, 2);
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
    AddMerge<> module;
    const size_t numMergeModules = math::RandInt(2, 10);
    for (size_t m = 0; m < numMergeModules; ++m)
    {
      IdentityLayer<> identityLayer;
      identityLayer.Forward(std::move(input),
          std::move(identityLayer.OutputParameter()));

      module.Add(identityLayer);
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
 * LSTM layer numerically gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientLSTMLayerTest)
{
  // LSTM function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(5, 1);
      target = arma::mat("1; 1; 1; 1; 1");
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(input, target, rho);
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
      arma::mat output;
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    RNN<NegativeLogLikelihood<> >* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
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
      input = arma::randu(5, 1);
      target = arma::mat("1; 1; 1; 1; 1");
      const size_t rho = 5;

      model = new RNN<NegativeLogLikelihood<> >(input, target, rho);
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
    arma::mat input, target;
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
      moduleA.Parameters().submat(100, 0, moduleA.Parameters().n_elem - 1, 0)),
      arma::accu(output.col(0)), 1e-3);

  BOOST_REQUIRE_CLOSE(arma::accu(
      moduleB.Parameters().submat(100, 0, moduleB.Parameters().n_elem - 1, 0)),
      arma::accu(output.col(1)), 1e-3);

  // Test the Backward function.
  error = arma::zeros(10, 2);
  module.Backward(std::move(input), std::move(error), std::move(delta));
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0);
}

/**
 * Concat layer numerically gradient test.
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

      model = new FFN<NegativeLogLikelihood<>, NguyenWidrowInitialization>(
          input, target);
      model->Add<IdentityLayer<> >();

      concat = new Concat<>();
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
      arma::mat output;
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
 * Simple test for the cross-entropy error performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleCrossEntropyErrorLayerTest)
{
  arma::mat input1, input2, output, target1, target2;
  CrossEntropyError<> module(1e-6);

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module.Forward(std::move(input1), std::move(target1));
  BOOST_REQUIRE_SMALL(error1 - 8 * std::log(2), 2e-5);

  input2 = arma::mat("0 1 1 0 1 0 0 1");
  target2 = arma::mat("0 1 1 0 1 0 0 1");
  double error2 = module.Forward(std::move(input2), std::move(target2));
  BOOST_REQUIRE_SMALL(error2, 1e-5);

  // Test the Backward function.
  module.Backward(std::move(input1), std::move(target1), std::move(output));
  for (double el : output)
  {
    // For the 0.5 constant vector we should get 1 / (1 - 0.5) = 2 everywhere.
    BOOST_REQUIRE_SMALL(el - 2, 5e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input1.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input1.n_cols);

  module.Backward(std::move(input2), std::move(target2), std::move(output));
  for (size_t i = 0; i < 8; ++i)
  {
    double el = output.at(0, i);
    if (input2.at(i) == 0)
      BOOST_REQUIRE_SMALL(el - 1, 2e-6);
    else
      BOOST_REQUIRE_SMALL(el + 1, 2e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input2.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input2.n_cols);
}

/*
 * Simple test for the mean squared error performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleMeanSquaredErrorLayerTest)
{
  arma::mat input, output, target;
  MeanSquaredError<> module;

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input = arma::mat("1.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0");
  target = arma::zeros(1, 8);
  double error = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_EQUAL(error, 0.5);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(target), std::move(output));
  // We subtract a zero vector, so the output should be equal with the input.
  CheckMatrices(input, output);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the error function on a single input.
  input = arma::mat("2");
  target = arma::mat("3");
  error = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_EQUAL(error, 1.0);

  // Test the Backward function on a single input.
  module.Backward(std::move(input), std::move(target), std::move(output));
  // Test whether the output is negative.
  BOOST_REQUIRE_EQUAL(arma::accu(output), -1);
  BOOST_REQUIRE_EQUAL(output.n_elem, 1);
}


BOOST_AUTO_TEST_SUITE_END();
