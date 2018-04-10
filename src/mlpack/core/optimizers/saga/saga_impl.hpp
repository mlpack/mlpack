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

#include <mlpack/core/optimizers/function.hpp>

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
    DecomposableFunctionType& function_,
    arma::mat& iterate)
{
  typedef Function<DecomposableFunctionType> FullFunctionType;
  FullFunctionType& function(static_cast<FullFunctionType&>(function_));

  // Make sure we have all the methods that we need.
  traits::CheckDecomposableFunctionTypeAPI<FullFunctionType>();
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  double overallObjective = 0;
  double lastObjective = DBL_MAX;
  size_t currentFunction = 0;
  size_t batch = 0;
  size_t batchStart = 0;

  // Initialize the update policy.
  if (resetPolicy)
    updatePolicy.Initialize(iterate.n_rows, iterate.n_cols);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat iterate0;

  // Find the number of batches.
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last batch.

  const size_t actualMaxIterations = (maxIterations == 0) ?
               std::numeric_limits<size_t>::max() : maxIterations;

  // initialize the gradients.
  size_t effectiveBatchSize = std::min(batchSize, numFunctions);
  arma::cube tableOfGradients(iterate.n_rows, iterate.n_cols, numBatches);
  arma::mat avgGradient(iterate.n_rows, iterate.n_cols);

  for (size_t f = 0, b = 0; f < numFunctions;
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
  avgGradient /= (double) numBatches; // Calculate average gradient

  for (size_t i = 0; i < actualMaxIterations; /* incrementing done manually */)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
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

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        function.Shuffle();
    }

    // Store current parameter for the calculation of the variance reduced
    // gradient.
    iterate0 = iterate;

    batch = math::RandInt(0, numBatches); // Select a random batch

    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of iterations left
    //       before actualMaxIterations is hit;
    // - the batch size can't be larger than the number of functions left.
    effectiveBatchSize = std::min(
      std::min(batchSize, actualMaxIterations - i),
      numFunctions - currentFunction);

    batchStart = batch * batchSize;

    // Calculate the gradient of a random function.
    function.Gradient(iterate, batchStart, gradient,
                      effectiveBatchSize);

    // Use the update policy to take a step.
    updatePolicy.Update(iterate, avgGradient, gradient,
                        tableOfGradients.slice(batch),
                        stepSize, numBatches);
    // Update the average gradient.
    avgGradient += (gradient-tableOfGradients.slice(batch))/numBatches;

    // Update the table of gradients.
    tableOfGradients.slice(batch) = gradient;

    overallObjective += function.Evaluate(iterate, currentFunction,
                                          effectiveBatchSize);
    currentFunction += effectiveBatchSize;
    i += effectiveBatchSize;
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
