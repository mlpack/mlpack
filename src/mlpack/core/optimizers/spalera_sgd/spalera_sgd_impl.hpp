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
SPALeRASGD<DecayPolicyType>::SPALeRASGD(const size_t batchSize,
                                        const double stepSize,
                                        const size_t maxIterations,
                                        const double tolerance,
                                        const double lambda,
                                        const double alpha,
                                        const double epsilon,
                                        const double adaptRate,
                                        const bool shuffle,
                                        const DecayPolicyType& decayPolicy,
                                        const bool resetPolicy) :
    batchSize(batchSize),
    stepSize(stepSize),
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
  // Find the number of functions.
  const size_t numFunctions = function.NumFunctions();
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  // Batch visitation order.
  arma::Col<size_t> visitationOrder = arma::linspace<arma::Col<size_t>>(0,
      (numBatches - 1), numBatches);

  if (shuffle)
    visitationOrder = arma::shuffle(visitationOrder);

  // To keep track of where we are and how things are going.
  size_t currentBatch = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  double currentObjective = overallObjective / numFunctions;

  // Initialize the update policy.
  if (resetPolicy)
  {
    updatePolicy.Initialize(iterate.n_rows, iterate.n_cols,
        currentObjective * lambda);
  }

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i != maxIterations; ++i, ++currentBatch)
  {
    // Is this iteration the start of a sequence?
    if ((currentBatch % numBatches) == 0)
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
      currentBatch = 0;

      if (shuffle)
        visitationOrder = arma::shuffle(visitationOrder);
    }

    // Evaluate the gradient for this mini-batch.
    const size_t offset = batchSize * visitationOrder[currentBatch];
    function.Gradient(iterate, offset, gradient);
    if (visitationOrder[currentBatch] != numBatches - 1)
    {
      for (size_t j = 1; j < batchSize; ++j)
      {
        arma::mat funcGradient;
        function.Gradient(iterate, offset + j, funcGradient);
        gradient += funcGradient;
      }

      // Now update the iterate.
      if (!updatePolicy.Update(stepSize, currentObjective, batchSize,
          numFunctions, iterate, gradient))
      {
          Log::Warn << "SPALeRA SGD: converged to " << overallObjective << "; "
              << "terminating with failure.  Try a smaller step size?"
              << std::endl;
          return overallObjective;
      }

      // Add that to the overall objective function.
      currentObjective = function.Evaluate(iterate, offset);
      for (size_t j = 1; j < batchSize; ++j)
        currentObjective += function.Evaluate(iterate, offset + j);

      overallObjective += currentObjective;
      currentObjective /= batchSize;
    }
    else
    {
      // Handle last batch differently: it's not a full-size batch.
      const size_t lastBatchSize = numFunctions - offset - 1;
      for (size_t j = 1; j < lastBatchSize; ++j)
      {
        arma::mat funcGradient;
        function.Gradient(iterate, offset + j, funcGradient);
        gradient += funcGradient;
      }

      // Ensure the last batch size isn't zero, to avoid division by zero before
      // updating.
      bool learningRateCheck = true;
      if (lastBatchSize > 0)
      {
        // Now update the iterate.
        learningRateCheck = updatePolicy.Update(stepSize, currentObjective,
            lastBatchSize, numFunctions, iterate, gradient);
      }
      else
      {
        // Now update the iterate.
        learningRateCheck = updatePolicy.Update(stepSize, currentObjective, 1,
            numFunctions, iterate, gradient);
      }

      if (!learningRateCheck)
      {
        Log::Warn << "SPALeRA SGD: converged to " << overallObjective << "; "
            << "terminating with failure.  Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      // Add that to the overall objective function.
      currentObjective += function.Evaluate(iterate, offset);
      for (size_t j = 1; j < lastBatchSize; ++j)
        currentObjective += function.Evaluate(iterate, offset + j);

      overallObjective += currentObjective;
      currentObjective /= lastBatchSize;
    }

    // Now update the learning rate if requested by the user.
    decayPolicy.Update(iterate, stepSize, gradient);
  }

  Log::Info << "SPALeRA SGD: maximum iterations (" << maxIterations << ") "
      << "reached; terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
