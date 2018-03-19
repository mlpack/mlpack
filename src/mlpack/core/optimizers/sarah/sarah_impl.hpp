/**
 * @file sarah_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of StochAstic Recusive gRadient algoritHm (SARAH).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SARAH_SARAH_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SARAH_SARAH_IMPL_HPP

// In case it hasn't been included yet.
#include "sarah.hpp"

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template<typename UpdatePolicyType>
SARAHType<UpdatePolicyType>::SARAHType(
    const double stepSize,
    const size_t batchSize,
    const size_t maxIterations,
    const size_t innerIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    innerIterations(innerIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    updatePolicy(updatePolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename UpdatePolicyType>
template<typename DecomposableFunctionType>
double SARAHType<UpdatePolicyType>::Optimize(
    DecomposableFunctionType& function, arma::mat& iterate)
{
  traits::CheckDecomposableFunctionTypeAPI<DecomposableFunctionType>();

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Set epoch length to n / b if the user asked for.
  if (innerIterations == 0)
    innerIterations = numFunctions;

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat v(iterate.n_rows, iterate.n_cols);
  arma::mat gradient0(iterate.n_rows, iterate.n_cols);
  arma::mat iterate0;

  // Find the number of batches.
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; ++i)
  {
    // Calculate the objective function.
    overallObjective = 0;
    for (size_t f = 0; f < numFunctions; f += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
      overallObjective += function.Evaluate(iterate, f, effectiveBatchSize);
    }

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Log::Warn << "SARAH: converged to " << overallObjective
          << "; terminating  with failure.  Try a smaller step size?"
          << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Log::Info << "SARAH: minimized within tolerance " << tolerance
          << "; terminating optimization." << std::endl;
      return overallObjective;
    }

    lastObjective = overallObjective;

    // Compute the full gradient.
    size_t effectiveBatchSize = std::min(batchSize, numFunctions);
    function.Gradient(iterate, 0, v, effectiveBatchSize);
    for (size_t f = effectiveBatchSize; f < numFunctions;
        /* incrementing done manually */)
    {
      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - f);

      function.Gradient(iterate, f, gradient, effectiveBatchSize);
      v += gradient;

      f += effectiveBatchSize;
    }
    v /= (double) numFunctions;

    // Update iterate with full gradient (v).
    iterate -= stepSize * v;

    const double vNorm = arma::norm(v);

    for (size_t f = 0, currentFunction = 0; f < innerIterations;
        /* incrementing done manually */)
    {
      // Is this iteration the start of a sequence?
      if ((currentFunction % numFunctions) == 0)
      {
        currentFunction = 0;

        // Determine order of visitation.
        if (shuffle)
          function.Shuffle();
      }

      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - currentFunction);

      // Calculate variance reduced gradient.
      function.Gradient(iterate, currentFunction, gradient,
          effectiveBatchSize);

      // Avoid an unnecessary copy on the first iteration.
      if (f > 0)
      {
        function.Gradient(iterate0, currentFunction, gradient0,
            effectiveBatchSize);

        // Store current parameter for the calculation of the variance reduced
        // gradient.
        iterate0 = iterate;

        // Use the update policy to take a step.
        if (updatePolicy.Update(iterate, v, gradient, gradient0,
            effectiveBatchSize, stepSize, vNorm))
        {
          break;
        }
      }
      else
      {
        // Store current parameter for the calculation of the variance reduced
        // gradient.
        iterate0 = iterate;

        // Use the update policy to take a step.
        if (updatePolicy.Update(iterate, v, gradient, gradient,
            effectiveBatchSize, stepSize, vNorm))
        {
          break;
        }
      }

      currentFunction += effectiveBatchSize;
      f += effectiveBatchSize;
    }
  }

  Log::Info << "SARAH: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
    overallObjective += function.Evaluate(iterate, i, effectiveBatchSize);
  }
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
