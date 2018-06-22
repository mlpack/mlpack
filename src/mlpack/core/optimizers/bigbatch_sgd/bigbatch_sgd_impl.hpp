/**
 * @file bigbatch_sgd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of big-batch SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BIGBATCH_SGD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BIGBATCH_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "bigbatch_sgd.hpp"

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template<typename UpdatePolicyType>
BigBatchSGD<UpdatePolicyType>::BigBatchSGD(
    const size_t batchSize,
    const double stepSize,
    const double batchDelta,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle) :
    batchSize(batchSize),
    stepSize(stepSize),
    batchDelta(batchDelta),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    updatePolicy(UpdatePolicyType())
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename UpdatePolicyType>
template<typename DecomposableFunctionType>
double BigBatchSGD<UpdatePolicyType>::Optimize(
    DecomposableFunctionType& function, arma::mat& iterate)
{
  typedef Function<DecomposableFunctionType> FullFunctionType;
  FullFunctionType& f(static_cast<FullFunctionType&>(function));

  // Make sure we have all the methods that we need.
  traits::CheckDecomposableFunctionTypeAPI<FullFunctionType>();


  // Find the number of functions to use.
  const size_t numFunctions = f.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;
  bool reset = false;
  arma::mat delta0, delta1;

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat functionGradient(iterate.n_rows, iterate.n_cols);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; /* incrementing done manually */)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0 && i > 0)
    {
      // Output current objective function.
      Log::Info << "Big-batch SGD: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "Big-batch SGD: converged to " << overallObjective
            << "; terminating with failure.  Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "Big-batch SGD: minimized within tolerance " << tolerance
            << "; terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        f.Shuffle();
    }

    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of iterations left
    //       before actualMaxIterations is hit;
    // - the batch size can't be larger than the number of functions left.
    size_t effectiveBatchSize = std::min(
        std::min(batchSize, actualMaxIterations - i),
        numFunctions - currentFunction);

    size_t k = 1;
    double vB = 0;

    // Compute the stochastic gradient estimation.
    f.Gradient(iterate, currentFunction, gradient, 1);

    delta1 = gradient;
    for (size_t j = 1; j < effectiveBatchSize; ++j, ++k)
    {
      f.Gradient(iterate, currentFunction + j, functionGradient, 1);
      delta0 = delta1 + (functionGradient - delta1) / k;

      // Compute sample variance.
      vB += arma::norm(functionGradient - delta1, 2.0) *
          arma::norm(functionGradient - delta0, 2.0);

      delta1 = delta0;
      gradient += functionGradient;
    }
    double gB = std::pow(arma::norm(gradient / effectiveBatchSize, 2), 2.0);

    // Reset the batch size update process counter.
    reset = false;

    // Increase batchSize only if there are more samples left.
    if (effectiveBatchSize == batchSize)
    {
      // Update batch size.
      while (gB <= ((1 / ((double) batchSize - 1) * vB) / batchSize))
      {
        // Increase batch size at least by one.
        size_t batchOffset = batchDelta * batchSize;
        if (batchOffset <= 0)
          batchOffset = 1;

        if ((currentFunction + batchSize + batchOffset) >= numFunctions)
          break;

        // Update the stochastic gradient estimation.
        const size_t batchStart = (currentFunction + batchSize + batchOffset
            - 1) < numFunctions ? currentFunction + batchSize - 1 : 0;
        for (size_t j = 0; j < batchOffset; ++j, ++k)
        {
          f.Gradient(iterate, batchStart + j, functionGradient, 1);
          delta0 = delta1 + (functionGradient - delta1) / (k + 1);

          // Compute sample variance.
          vB += arma::norm(functionGradient - delta1, 2.0) *
              arma::norm(functionGradient - delta0, 2.0);

          delta1 = delta0;
          gradient += functionGradient;
        }
        gB = std::pow(arma::norm(gradient / (batchSize + batchOffset), 2), 2.0);

        // Update the batchSize.
        batchSize += batchOffset;
        effectiveBatchSize += batchOffset;

        // Batch size updated.
        reset = true;
      }
    }

    updatePolicy.Update(f, stepSize, iterate, gradient, gB, vB,
        currentFunction, batchSize, effectiveBatchSize, reset);

    // Update the iterate.
    iterate -= stepSize * gradient;

    overallObjective += f.Evaluate(iterate, currentFunction,
        effectiveBatchSize);

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;
  }

  Log::Info << "Big-batch SGD: maximum iterations (" << maxIterations << ") "
      << "reached; terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
    overallObjective += f.Evaluate(iterate, i, effectiveBatchSize);
  }
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
