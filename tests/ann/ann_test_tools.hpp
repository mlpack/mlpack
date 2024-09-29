/**
 * @file tests/ann_test_tools.hpp
 * @author Marcus Edel
 *
 * This file includes some useful functions for ann tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_ANN_TEST_TOOLS_HPP
#define MLPACK_TESTS_ANN_TEST_TOOLS_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

using namespace mlpack;

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
  output.set_size(module.OutputSize(), input.n_cols);
  outputA.set_size(module.OutputSize(), input.n_cols);
  outputB.set_size(module.OutputSize(), input.n_cols);

  // Initialize the input matrix.
  RandomInitialization init(minValue, maxValue);
  init.Initialize(input, input.n_rows, input.n_cols);

  // Initialize the jacobian matrix.
  module.Forward(input, output);
  jacobianA = arma::zeros(input.n_elem, output.n_elem);

  // Share the input paramter matrix.
  arma::mat sin = arma::mat(input.memptr(), input.n_rows, input.n_cols,
      false, false);

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    double original = sin(i);
    sin(i) = original - perturbation;
    module.Forward(input, outputA);
    sin(i) = original + perturbation;
    module.Forward(input, outputB);
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

    arma::mat delta(input.n_rows, input.n_cols);
    module.Backward(input, output, deriv, delta);

    jacobianB.col(i) = delta;
  }

  return arma::max(arma::max(arma::abs(jacobianA - jacobianB)));
}

// Custom Jacobian Test where we get the input from outside of this function
// unlike the original Jacobian Test where input is generated inside that
// funcion.
template <typename ModuleType>
double CustomJacobianTest(ModuleType& module,
                          arma::mat& input,
                          const double perturbation = 1e-6)
{
  arma::mat output, outputA, outputB, jacobianA, jacobianB;
  output.set_size(module.OutputSize(), input.n_cols);
  outputA.set_size(module.OutputSize(), input.n_cols);
  outputB.set_size(module.OutputSize(), input.n_cols);

  // Initialize the jacobian matrix.
  module.Forward(input, output);
  jacobianA = arma::zeros(input.n_elem, output.n_elem);

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    double original = input(i);
    input(i) = original - perturbation;
    module.Forward(input, outputA);
    input(i) = original + perturbation;
    module.Forward(input, outputB);
    input(i) = original;

    outputB -= outputA;
    outputB /= 2 * perturbation;
    jacobianA.row(i) = outputB.t();
  }

  // Initialize the derivative parameter.
  arma::mat deriv = arma::zeros(output.n_rows, output.n_cols);

  // Initialize the jacobian matrix.
  jacobianB = arma::zeros(input.n_elem, output.n_elem);

  for (size_t i = 0; i < deriv.n_elem; ++i)
  {
    deriv.zeros();
    deriv(i) = 1;

    arma::mat delta(input.n_rows, input.n_cols);
    module.Backward(input, output, deriv, delta);

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
  module.Forward(input, target);

  arma::mat delta;
  module.Backward(input, target, delta);

  arma::mat centralDifference = arma::zeros(delta.n_rows, delta.n_cols);
  arma::mat inputTemp = arma::mat(input.memptr(), input.n_rows, input.n_cols,
      false, false);

  arma::mat centralDifferenceTemp = arma::mat(centralDifference.memptr(),
      centralDifference.n_rows, centralDifference.n_cols, false, false);

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    inputTemp(i) = inputTemp(i) + eps;
    double outputA = module.Forward(input, target);
    inputTemp(i) = inputTemp(i) - (2 * eps);
    double outputB = module.Forward(input, target);

    centralDifferenceTemp(i) = (outputA - outputB) / (2 * eps);
    inputTemp(i) = inputTemp(i) + eps;
  }

  return arma::max(arma::max(arma::abs(centralDifference - delta)));
}

template<typename ActivationFunction>
double ActivationJacobianTest(arma::mat& input,
                              const double minValue = -2,
                              const double maxValue = -1,
                              const double pertubation = 1e-6)
{
  arma::mat output, outputA, outputB, jacobianA, jacobianB;
  output.set_size(input.n_rows, input.n_cols);
  outputA.set_size(input.n_rows, input.n_cols);
  outputB.set_size(input.n_rows, input.n_cols);

  RandomInitialization init(minValue, maxValue);
  init.Initialize(input, input.n_rows, input.n_cols);

  ActivationFunction::Fn(input, output);
  jacobianA = arma::zeros(input.n_elem, output.n_elem);

  for(size_t i = 0; i < input.n_elem; ++i)
  {
    double original = input(i);
    input(i) = original - pertubation;
    ActivationFunction::Fn(input, outputA);
    input(i) = original + pertubation;
    ActivationFunction::Fn(input, outputB);
    input(i) = original;

    outputB -= outputA;
    outputB /= 2 * pertubation;
    jacobianA.row(i) = outputB.t();
  }

  arma::mat deriv = arma::zeros(output.n_rows, output.n_cols);
  jacobianB = arma::zeros(input.n_elem, output.n_elem);

  for(size_t i = 0; i < deriv.n_elem; ++i)
  {
    deriv.zeros();
    deriv(i) = 1;

    arma::mat delta(input.n_rows, input.n_cols);
    ActivationFunction::Deriv(input, output, delta);
    delta %= deriv;

    jacobianB.col(i) = delta;
  }

  return arma::max(arma::max(arma::abs(jacobianA - jacobianB)));
}

// Simple numerical gradient checker.
template<class FunctionType, typename MatType = arma::mat>
double CheckGradient(FunctionType& function, const double eps = 1e-7)
{
  // Get gradients for the current parameters.
  MatType orgGradient, gradient, estGradient;
  function.Gradient(orgGradient);

  estGradient = arma::zeros<MatType>(orgGradient.n_rows, orgGradient.n_cols);

  // Compute numeric approximations to gradient.
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

// Simple numerical gradient checker for regularizers.
template<class FunctionType>
double CheckRegularizerGradient(FunctionType& function, const double eps = 1e-7)
{
  // Get gradients for the current parameters.
  arma::mat weight = arma::randu(10, 10);
  arma::mat orgGradient = arma::zeros(10 * 10, 1);
  function.Gradient(weight, orgGradient);

  arma::mat estGradient = arma::zeros(weight.n_rows, weight.n_cols);

  // Compute numeric approximations to gradient.
  for (size_t i = 0; i < weight.n_rows; ++i)
  {
    for (size_t j = 0; j < weight.n_cols; ++j)
    {
      double tmp = weight(i, j);

      weight(i, j) += eps;
      double costPlus = function.Output(weight, i, j);
      weight(i, j) -= (2 * eps);
      double costMinus = function.Output(weight, i, j);

      // Restore the weight value.
      weight(i, j) = tmp;
      estGradient(i, j) = (costPlus - costMinus) / (2 * eps);
    }
  }

  estGradient = vectorise(estGradient);
  // Estimate error of gradient.
  return arma::norm(orgGradient - estGradient) /
      arma::norm(orgGradient + estGradient);
}

#endif
