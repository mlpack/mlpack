/**
 * @file spalera_sgd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of SPALeRA SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SPALERA_SGD_SPALERA_SGD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SPALERA_SGD_SPALERA_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "spalera_sgd.hpp"

namespace mlpack {
namespace optimization {

template<typename DecayPolicyType>
SPALeRASGD<DecayPolicyType>::SPALeRASGD(const double stepSize,
                                        const size_t batchSize,
                                        const size_t maxIterations,
                                        const double tolerance,
                                        const double lambda,
                                        const double alpha,
                                        const double epsilon,
                                        const double adaptRate,
                                        const bool shuffle,
                                        const DecayPolicyType& decayPolicy,
                                        const bool resetPolicy) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    lambda(lambda),
    shuffle(shuffle),
    updatePolicy(SPALeRAStepsize(alpha, epsilon, adaptRate)),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecayPolicyType>
template<typename DecomposableFunctionType>
double SPALeRASGD<DecayPolicyType>::Optimize(DecomposableFunctionType& function,
                                             arma::mat& iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
    overallObjective += function.Evaluate(iterate, i, effectiveBatchSize);
  }

  double currentObjective = overallObjective / numFunctions;

  // Initialize the update policy.
  if (resetPolicy)
  {
    updatePolicy.Initialize(iterate.n_rows, iterate.n_cols,
        currentObjective * lambda);
  }

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; /* incrementing done manually */)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      // Output current objective function.
      Log::Info << "SPALeRA SGD: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "SPALeRA SGD: converged to " << overallObjective
            << "; terminating with failure.  Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "SPALeRA SGD: minimized within tolerance " << tolerance
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

    // Find the effective batch size (the last batch may be smaller).
    const size_t effectiveBatchSize = std::min(batchSize,
        numFunctions - currentFunction);

    function.Gradient(iterate, currentFunction, gradient, effectiveBatchSize);

    // Use the update policy to take a step.
    if (!updatePolicy.Update(stepSize, currentObjective, effectiveBatchSize,
        numFunctions, iterate, gradient))
    {
        Log::Warn << "SPALeRA SGD: converged to " << overallObjective << "; "
            << "terminating with failure.  Try a smaller step size?"
            << std::endl;
        return overallObjective;
    }

    currentObjective = function.Evaluate(iterate, currentFunction,
        effectiveBatchSize);

    // Now update the learning rate if requested by the user.
    decayPolicy.Update(iterate, stepSize, gradient);

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;
    overallObjective += currentObjective;
    currentObjective /= effectiveBatchSize;
  }

  Log::Info << "SPALeRA SGD: maximum iterations (" << maxIterations
      << ") reached; terminating optimization." << std::endl;

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
