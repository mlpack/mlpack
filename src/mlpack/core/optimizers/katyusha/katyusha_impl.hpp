/**
 * @file katyusha_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of Katyusha a direct, primal-only stochastic gradient method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_KATYUSHA_KATYUSHA_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_KATYUSHA_KATYUSHA_IMPL_HPP

// In case it hasn't been included yet.
#include "katyusha.hpp"

namespace mlpack {
namespace optimization {

Katyusha::Katyusha(
    const double stepSize,
    const double lambda,
    const double tau,
    const size_t batchSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle) :
    stepSize(stepSize),
    lambda(lambda),
    tau(tau),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double Katyusha::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();
  const double tau1 = 0.5 - tau;
  const double tau2 = 1 + lambda * stepSize;

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

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat fullGradient(iterate.n_rows, iterate.n_cols);
  arma::mat gradient0(iterate.n_rows, iterate.n_cols);
  arma::mat iterate0;

  arma::mat y = arma::zeros<arma::mat>(iterate.n_rows, iterate.n_cols);
  arma::mat z = arma::zeros<arma::mat>(iterate.n_rows, iterate.n_cols);
  arma::mat w = arma::zeros<arma::mat>(iterate.n_rows, iterate.n_cols);

  // Find the number of batches.
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; /* incrementing done manually */)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      // Output current objective function.
      Log::Info << "Katyusha: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "Katyusha: converged to " << overallObjective
            << "; terminating  with failure.  Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "Katyusha: minimized within tolerance " << tolerance
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

    iterate0 = iterate;
    y = iterate;
    z = iterate;
    w.zeros();

    // Compute the full gradient.
    size_t effectiveBatchSize = std::min(batchSize, numFunctions);
    function.Gradient(iterate, 0, fullGradient, effectiveBatchSize);
    for (size_t f = effectiveBatchSize; f < numFunctions;
        /* incrementing done manually */)
    {
      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - f);

      function.Gradient(iterate, f, gradient, effectiveBatchSize);
      fullGradient += gradient;

      f += effectiveBatchSize;
    }
    fullGradient /= (double) numFunctions;

    // To keep track of where we are and how things are going.
    double bb = 0;
    size_t iteration = 0;

    for (size_t f = 0; f < numFunctions; /* incrementing done manually */)
    {
      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - f);

      iterate = tau * z + 0.5 * iterate0 + y * tau1;

      // Calculate variance reduced gradient.
      function.Gradient(iterate, f, gradient, effectiveBatchSize);
      function.Gradient(iterate0, f, gradient0, effectiveBatchSize);

      const arma::mat z0 = z;
      z -= stepSize * (fullGradient + (gradient - gradient0) /
        (double) batchSize);
      y = iterate + (z - z0) * tau;
      w += y * std::pow(tau2, (double) iteration);

      bb += pow(tau2, iteration);

      f += effectiveBatchSize;

      iteration++;
    }

    iterate = w / (double) bb;

    // Find the effective batch size (the last batch may be smaller).
    effectiveBatchSize = std::min(batchSize, numFunctions - currentFunction);

    overallObjective += function.Evaluate(iterate, currentFunction,
        effectiveBatchSize);

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;
  }

  Log::Info << "Katyusha: maximum iterations (" << maxIterations << ") reached"
      << "; terminating optimization." << std::endl;

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
