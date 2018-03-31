/**
 * @file aug_lagrangian_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of AugLagrangian class (Augmented Lagrangian optimization
 * method).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP

#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include <mlpack/core/optimizers/function.hpp>
#include "aug_lagrangian_function.hpp"

namespace mlpack {
namespace optimization {

template<typename LagrangianFunctionType>
bool AugLagrangian::Optimize(LagrangianFunctionType& function,
                             arma::mat& coordinates,
                             const arma::vec& initLambda,
                             const double initSigma,
                             const size_t maxIterations)
{
  lambda = initLambda;
  sigma = initSigma;

  AugLagrangianFunction<LagrangianFunctionType> augfunc(function,
      lambda, sigma);

  return Optimize(augfunc, coordinates, maxIterations);
}

template<typename LagrangianFunctionType>
bool AugLagrangian::Optimize(LagrangianFunctionType& function,
                             arma::mat& coordinates,
                             const size_t maxIterations)
{
  // If the user did not specify the right size for sigma and lambda, we will
  // use defaults.
  if (!lambda.is_empty())
  {
    AugLagrangianFunction<LagrangianFunctionType> augfunc(function, lambda,
        sigma);
    return Optimize(augfunc, coordinates, maxIterations);
  }
  else
  {
    AugLagrangianFunction<LagrangianFunctionType> augfunc(function);
    return Optimize(augfunc, coordinates, maxIterations);
  }
}

template<typename LagrangianFunctionType>
bool AugLagrangian::Optimize(
    AugLagrangianFunction<LagrangianFunctionType>& augfunc,
    arma::mat& coordinates,
    const size_t maxIterations)
{
  traits::CheckConstrainedFunctionTypeAPI<LagrangianFunctionType>();

  LagrangianFunctionType& function = augfunc.Function();

  // Ensure that we update lambda immediately.
  double penaltyThreshold = DBL_MAX;

  // Track the last objective to compare for convergence.
  double lastObjective = function.Evaluate(coordinates);

  // Then, calculate the current penalty.
  double penalty = 0;
  for (size_t i = 0; i < function.NumConstraints(); i++)
    penalty += std::pow(function.EvaluateConstraint(i, coordinates), 2);

  Log::Debug << "Penalty is " << penalty << " (threshold " << penaltyThreshold
      << ")." << std::endl;

  // The odd comparison allows user to pass maxIterations = 0 (i.e. no limit on
  // number of iterations).
  size_t it;
  for (it = 0; it != (maxIterations - 1); it++)
  {
    Log::Info << "AugLagrangian on iteration " << it
        << ", starting with objective "  << lastObjective << "." << std::endl;

    if (!lbfgs.Optimize(augfunc, coordinates))
      Log::Info << "L-BFGS reported an error during optimization."
          << std::endl;

    // Check if we are done with the entire optimization (the threshold we are
    // comparing with is arbitrary).
    if (std::abs(lastObjective - function.Evaluate(coordinates)) < 1e-10 &&
        augfunc.Sigma() > 500000)
    {
      lambda = std::move(augfunc.Lambda());
      sigma = augfunc.Sigma();
      return true;
    }

    lastObjective = function.Evaluate(coordinates);

    // Assuming that the optimization has converged to a new set of coordinates,
    // we now update either lambda or sigma.  We update sigma if the penalty
    // term is too high, and we update lambda otherwise.

    // First, calculate the current penalty.
    double penalty = 0;
    for (size_t i = 0; i < function.NumConstraints(); i++)
    {
      penalty += std::pow(function.EvaluateConstraint(i, coordinates), 2);
    }

    Log::Info << "Penalty is " << penalty << " (threshold "
        << penaltyThreshold << ")." << std::endl;

    if (penalty < penaltyThreshold) // We update lambda.
    {
      // We use the update: lambda_{k + 1} = lambda_k - sigma * c(coordinates),
      // but we have to write a loop to do this for each constraint.
      for (size_t i = 0; i < function.NumConstraints(); i++)
        augfunc.Lambda()[i] -= augfunc.Sigma() *
            function.EvaluateConstraint(i, coordinates);

      // We also update the penalty threshold to be a factor of the current
      // penalty.  TODO: this factor should be a parameter (from CLI).  The
      // value of 0.25 is taken from Burer and Monteiro (2002).
      penaltyThreshold = 0.25 * penalty;
      Log::Info << "Lagrange multiplier estimates updated." << std::endl;
    }
    else
    {
      // We multiply sigma by a constant value.  TODO: this factor should be a
      // parameter (from CLI).  The value of 10 is taken from Burer and Monteiro
      // (2002).
      augfunc.Sigma() *= 10;
      Log::Info << "Updated sigma to " << augfunc.Sigma() << "." << std::endl;
    }
  }

  return false;
}

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP

