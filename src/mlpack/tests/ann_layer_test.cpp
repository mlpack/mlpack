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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNLayerTest);

// Helper function whcih calls the Reset function of the given module.
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

    centralDifferenceTemp(i) = (outputA - outputB) / ( 2 * eps);
    inputTemp(i) = inputTemp(i) + eps;
  }

  return arma::max(arma::max(arma::abs(centralDifference - delta)));
}

/**
 * Simple add module test.
 */
BOOST_AUTO_TEST_CASE(SimpleAddLayerTest)
{
  arma::mat output, input, delta;
  Add<> module(10);

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

BOOST_AUTO_TEST_SUITE_END();