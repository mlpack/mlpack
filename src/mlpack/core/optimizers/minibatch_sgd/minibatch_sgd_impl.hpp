/**
 * @file minibatch_sgd_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of mini-batch SGD.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_MINIBATCH_SGD_MINIBATCH_SGD_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_MINIBATCH_SGD_MINIBATCH_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "minibatch_sgd.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
MiniBatchSGD<DecomposableFunctionType>::MiniBatchSGD(
    DecomposableFunctionType& function,
    const size_t batchSize,
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle) :
    function(function),
    batchSize(batchSize),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double MiniBatchSGD<DecomposableFunctionType>::Optimize(arma::mat& iterate)
{
  // Find the number of functions.
  const size_t numFunctions = function.NumFunctions();
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.
  std::cout << "numBatches " << numBatches << ".\n";

  // This is only used if shuffle is true.
  arma::Col<size_t> visitationOrder;
  if (shuffle)
    visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
        (numBatches - 1), numBatches));

  // To keep track of where we are and how things are going.
  size_t currentBatch = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i != maxIterations; ++i, ++currentBatch)
  {
    // Is this iteration the start of a sequence?
    if ((currentBatch % numBatches) == 0)
    {
      // Output current objective function.
      std::cout << "Mini-batch SGD: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "Mini-batch SGD: converged to " << overallObjective
            << "; terminating with failure.  Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "Mini-batch SGD: minimized within tolerance " << tolerance
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
    const size_t offset = (shuffle) ? batchSize * visitationOrder[currentBatch]
        : batchSize * currentBatch;
    function.Gradient(iterate, offset, gradient);
    for (size_t j = 1; j < batchSize; ++j)
    {
      arma::mat funcGradient;
      function.Gradient(iterate, offset + j, funcGradient);
      gradient += funcGradient;
    }

    // Now update the iterate.
    iterate -= (stepSize / batchSize) * gradient;

    // Add that to the overall objective function.
    for (size_t j = 0; j < batchSize; ++j)
      overallObjective += function.Evaluate(iterate, offset + j);
  }

  Log::Info << "Mini-batch SGD: maximum iterations (" << maxIterations << ") "
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
