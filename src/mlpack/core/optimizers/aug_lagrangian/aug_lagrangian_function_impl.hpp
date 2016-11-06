/**
 * @file aug_lagrangian_function_impl.hpp
 * @author Ryan Curtin
 *
 * Simple, naive implementation of AugLagrangianFunction.  Better
 * specializations can probably be given in many cases, but this is the most
 * general case.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_FUNCTION_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_FUNCTION_IMPL_HPP

// In case it hasn't been included.
#include "aug_lagrangian_function.hpp"

namespace mlpack {
namespace optimization {

// Initialize the AugLagrangianFunction.
template<typename LagrangianFunction>
AugLagrangianFunction<LagrangianFunction>::AugLagrangianFunction(
    LagrangianFunction& function) :
    function(function),
    lambda(function.NumConstraints()),
    sigma(10)
{
  // Initialize lambda vector to all zeroes.
  lambda.zeros();
}

// Initialize the AugLagrangianFunction.
template<typename LagrangianFunction>
AugLagrangianFunction<LagrangianFunction>::AugLagrangianFunction(
    LagrangianFunction& function,
    const arma::vec& lambda,
    const double sigma) :
    lambda(lambda),
    sigma(sigma),
    function(function)
{
  // Nothing else to do.
}

// Evaluate the AugLagrangianFunction at the given coordinates.
template<typename LagrangianFunction>
double AugLagrangianFunction<LagrangianFunction>::Evaluate(
    const arma::mat& coordinates) const
{
  // The augmented Lagrangian is evaluated as
  //    f(x) + {-lambda_i * c_i(x) + (sigma / 2) c_i(x)^2} for all constraints

  // First get the function's objective value.
  double objective = function.Evaluate(coordinates);

  // Now loop for each constraint.
  for (size_t i = 0; i < function.NumConstraints(); ++i)
  {
    double constraint = function.EvaluateConstraint(i, coordinates);

    objective += (-lambda[i] * constraint) +
        sigma * std::pow(constraint, 2) / 2;
  }

  return objective;
}

// Evaluate the gradient of the AugLagrangianFunction at the given coordinates.
template<typename LagrangianFunction>
void AugLagrangianFunction<LagrangianFunction>::Gradient(
    const arma::mat& coordinates,
    arma::mat& gradient) const
{
  // The augmented Lagrangian's gradient is evaluted as
  // f'(x) + {(-lambda_i + sigma * c_i(x)) * c'_i(x)} for all constraints
  gradient.zeros();
  function.Gradient(coordinates, gradient);

  arma::mat constraintGradient; // Temporary for constraint gradients.
  for (size_t i = 0; i < function.NumConstraints(); i++)
  {
    function.GradientConstraint(i, coordinates, constraintGradient);

    // Now calculate scaling factor and add to existing gradient.
    arma::mat tmpGradient;
    tmpGradient = (-lambda[i] + sigma *
        function.EvaluateConstraint(i, coordinates)) * constraintGradient;
    gradient += tmpGradient;
  }
}

// Get the initial point.
template<typename LagrangianFunction>
const arma::mat& AugLagrangianFunction<LagrangianFunction>::GetInitialPoint()
    const
{
  return function.GetInitialPoint();
}

} // namespace optimization
} // namespace mlpack

#endif

