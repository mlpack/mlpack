/**
 * @file scd_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of stochastic coordinate descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SCD_SCD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SCD_SCD_IMPL_HPP

// In case it hasn't been included yet.
#include "scd.hpp"

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template <typename DescentPolicyType>
SCD<DescentPolicyType>::SCD(
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const size_t updateInterval,
    const DescentPolicyType descentPolicy) :
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    updateInterval(updateInterval),
    descentPolicy(descentPolicy)
{ /* Nothing to do */ }

//! Optimize the function (minimize).
template <typename DescentPolicyType>
template <typename ResolvableFunctionType>
double SCD<DescentPolicyType>::Optimize(ResolvableFunctionType& function,
                                        arma::mat& iterate)
{
  // Make sure we have the methods that we need.
  traits::CheckResolvableFunctionTypeAPI<ResolvableFunctionType>();

  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  arma::sp_mat gradient;

  // Start iterating.
  for (size_t i = 1; i != maxIterations; ++i)
  {
    // Get the coordinate to descend on.
    size_t featureIdx = descentPolicy.DescentFeature(i, iterate, function);

    // Get the partial gradient with respect to this feature.
    function.PartialGradient(iterate, featureIdx, gradient);

    // Update the decision variable with the partial gradient.
    iterate.col(featureIdx) -= stepSize * gradient.col(featureIdx);

    // Check for convergence.
    if (i % updateInterval == 0)
    {
      overallObjective = function.Evaluate(iterate);

      // Output current objective function.
      Log::Info << "SCD: iteration " << i << ", objective " << overallObjective
          << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "SCD: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "SCD: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
        return overallObjective;
      }

      lastObjective = overallObjective;
    }
  }

  Log::Info << "SCD: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate and return final objective.
  return function.Evaluate(iterate);
}

} // namespace optimization
} // namespace mlpack

#endif
