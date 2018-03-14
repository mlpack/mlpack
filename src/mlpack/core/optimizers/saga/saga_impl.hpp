/**
* @file saga_impl.hpp
* @author Prabhat Sharma
*
* Implementation of SAGA: A Fast Incremental Gradient Method With
* Support for Non-Strongly Convex Composite Objectives.
*
* mlpack is free software; you may redistribute it and/or modify it under the
* terms of the 3-clause BSD license.  You should have received a copy of the
* 3-clause BSD license along with mlpack.  If not, see
* http://www.opensource.org/licenses/BSD-3-Clause for more information.
*/
#ifndef MLPACK_CORE_OPTIMIZERS_SAGA_SAGA_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SAGA_SAGA_IMPL_HPP

// In case it hasn't been included yet.
#include "saga.hpp"

namespace mlpack {
namespace optimization {

template<typename UpdatePolicyType, typename DecayPolicyType>
SAGAType<UpdatePolicyType, DecayPolicyType>::SAGAType(
    const double stepSize,
    const size_t batchSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy,
    const DecayPolicyType& decayPolicy,
    const bool resetPolicy) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    updatePolicy(updatePolicy),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename UpdatePolicyType, typename DecayPolicyType>
template<typename DecomposableFunctionType>
double SAGAType<UpdatePolicyType, DecayPolicyType>::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Initialize the update policy.
  if (resetPolicy)
    updatePolicy.Initialize(iterate.n_rows, iterate.n_cols);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat gradient0(iterate.n_rows, iterate.n_cols);
  arma::mat iterate0;

  // Find the number of batches.
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  const size_t actualMaxIterations = (maxIterations == 0) ?
               std::numeric_limits<size_t>::max() : maxIterations;

  // initialize the gradients.
  size_t effectiveBatchSize = std::min(batchSize, numFunctions);
  arma::cube tableOfGradients(iterate.n_rows, iterate.n_cols, numBatches);
  arma::mat avgGradient(iterate.n_rows, iterate.n_cols);

  for (size_t f = 0, b=0; f < numFunctions;
    /* incrementing done manually */)
  {
    // Find the effective batch size (the last batch may be smaller).
    effectiveBatchSize = std::min(batchSize, numFunctions - f);

    function.Gradient(iterate, f, tableOfGradients.slice(b),
                      effectiveBatchSize);

    f += effectiveBatchSize;
    avgGradient += tableOfGradients.slice(b);
    b++;
  }
  avgGradient /= (double) numBatches;

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
      Log::Warn << "SAGA: converged to " << overallObjective
                << "; terminating  with failure.  Try a smaller step size?"
                << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Log::Info << "SAGA: minimized within tolerance " << tolerance
                << "; terminating optimization." << std::endl;
      return overallObjective;
    }

    lastObjective = overallObjective;

    // Store current parameter for the calculation of the variance reduced
    // gradient.
    iterate0 = iterate;

    for (size_t f = 0, b=0, currentFunction = 0; f < numFunctions;
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

      b = math::RandInt(0, numBatches); // Random batch selected
      // Find the effective batch size (the last batch may be smaller).
      currentFunction = b*batchSize;
      effectiveBatchSize = std::min(batchSize, numFunctions - currentFunction);

      // Calculate the gradient of a random function.
      function.Gradient(iterate, currentFunction, gradient,
                        effectiveBatchSize);
      gradient0 = tableOfGradients.slice(b);

      // Use the update policy to take a step.
      updatePolicy.Update(iterate, avgGradient, gradient, gradient0,
                          stepSize);
      // update average
      avgGradient += (gradient-tableOfGradients.slice(b))/numBatches;

      // update table of gradients
      tableOfGradients.slice(b) = gradient;

      f += effectiveBatchSize;
    }

    // Update the learning rate if requested by the user.
    decayPolicy.Update(iterate, iterate0, gradient, avgGradient, numBatches,
                       stepSize);
  }

  Log::Info << "SAGA: maximum iterations (" << maxIterations << ") reached; "
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
