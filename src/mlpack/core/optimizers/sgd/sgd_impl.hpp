/**
 * @file sgd_impl.hpp
 * @author Ryan Curtin
 * @author Arun Reddy
 * @author Abhinav Moudgil
 *
 * Implementation of stochastic gradient descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_SGD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_SGD_IMPL_HPP

#include <mlpack/methods/regularized_svd/regularized_svd_function.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>

// In case it hasn't been included yet.
#include "sgd.hpp"

namespace mlpack {
namespace optimization {

template<typename UpdatePolicyType>
SGD<UpdatePolicyType>::SGD(
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType updatePolicy,
    const bool resetPolicy) :
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    updatePolicy(updatePolicy),
    resetPolicy(resetPolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename UpdatePolicyType>
template<typename DecomposableFunctionType>
double SGD<UpdatePolicyType>::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // This is used only if shuffle is true.
  arma::Col<size_t> visitationOrder;
  if (shuffle)
  {
    visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
        (numFunctions - 1), numFunctions));
  }

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  // Initialize the update policy.
  if (resetPolicy)
    updatePolicy.Initialize(iterate.n_rows, iterate.n_cols);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i != maxIterations; ++i, ++currentFunction)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      // Output current objective function.
      Log::Info << "SGD: iteration " << i << ", objective " << overallObjective
          << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "SGD: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "SGD: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        visitationOrder = arma::shuffle(visitationOrder);
    }

    // Evaluate the gradient for this iteration.
    if (shuffle)
      function.Gradient(iterate, visitationOrder[currentFunction], gradient);
    else
      function.Gradient(iterate, currentFunction, gradient);

    // Use the update policy to take a step.
    updatePolicy.Update(iterate, stepSize, gradient);

    // Now add that to the overall objective function.
    if (shuffle)
    {
      overallObjective += function.Evaluate(iterate,
          visitationOrder[currentFunction]);
    }
    else
    {
      overallObjective += function.Evaluate(iterate, currentFunction);
    }
  }

  Log::Info << "SGD: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
