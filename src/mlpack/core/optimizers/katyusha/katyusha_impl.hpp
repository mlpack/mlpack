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

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template<bool Proximal>
KatyushaType<Proximal>::KatyushaType(
    const double convexity,
    const double lipschitz,
    const size_t batchSize,
    const size_t maxIterations,
    const size_t innerIterations,
    const double tolerance,
    const bool shuffle) :
    convexity(convexity),
    lipschitz(lipschitz),
    batchSize(batchSize),
    maxIterations(maxIterations),
    innerIterations(innerIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<bool Proximal>
template<typename DecomposableFunctionType>
double KatyushaType<Proximal>::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  traits::CheckDecomposableFunctionTypeAPI<DecomposableFunctionType>();

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // Set epoch length to n / b if the user asked for.
  if (innerIterations == 0)
    innerIterations = numFunctions;

  // Find the number of batches.
  size_t numBatches = innerIterations / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  const double tau1 = std::min(0.5,
      std::sqrt(batchSize * convexity / (3.0 * lipschitz)));
  const double tau2 = 0.5;
  const double alpha = 1.0 / (3.0 * tau1 * lipschitz);
  const double r = 1.0 + std::min(alpha * convexity, 1.0 /
      (4.0 / innerIterations));

  // sum_{j=0}^{m-1} 1 + std::min(alpha * convexity, 1 / (4 * m)^j).
  double normalizer = 1;
  for (size_t i = 0; i < numBatches; i++)
  {
    normalizer = r * (normalizer + 1.0);
  }
  normalizer = 1.0 / normalizer;

  // To keep track of where we are and how things are going.
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat fullGradient(iterate.n_rows, iterate.n_cols);
  arma::mat gradient0(iterate.n_rows, iterate.n_cols);

  arma::mat iterate0 = iterate;
  arma::mat y = iterate;
  arma::mat z = iterate;
  arma::mat w = arma::zeros<arma::mat>(iterate.n_rows, iterate.n_cols);

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; ++i)
  {
    // Calculate the objective function.
    overallObjective = 0;
    for (size_t f = 0; f < numFunctions; f += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
      overallObjective += function.Evaluate(iterate0, f, effectiveBatchSize);
    }

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

    lastObjective = overallObjective;

    // Compute the full gradient.
    size_t effectiveBatchSize = std::min(batchSize, numFunctions);
    function.Gradient(iterate, 0, fullGradient, effectiveBatchSize);
    for (size_t f = effectiveBatchSize; f < numFunctions;
        /* incrementing done manually */)
    {
      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - f);

      function.Gradient(iterate0, f, gradient, effectiveBatchSize);
      fullGradient += gradient;

      f += effectiveBatchSize;
    }
    fullGradient /= (double) numFunctions;

    // To keep track of where we are and how things are going.
    double cw = 1;
    w.zeros();

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
      iterate = tau1 * z + tau2 * iterate0 + (1 - tau1 - tau2) * y;

      // Calculate variance reduced gradient.
      function.Gradient(iterate, currentFunction, gradient,
          effectiveBatchSize);
      function.Gradient(iterate0, currentFunction, gradient0,
          effectiveBatchSize);

      // By the minimality definition of z_{k + 1}, we have that:
      // z_{k+1} − z_k + \alpha * \sigma_{k+1} + \alpha g = 0.
      arma::mat zNew = z - alpha * (fullGradient + (gradient - gradient0) /
          (double) batchSize);

      // Proximal update, choose between Option I and Option II. Shift relative
      // to the Lipschitz constant or take a constant step using the given step
      // size.
      if (Proximal)
      {
        // yk = x0 − 1 / (3L) * \delta1, k = 1
        // yk = x0 − 1 / (3L) * \delta2 - ((1 - tau) / (3L)) + tau * alpha)
        // * \delta1, k = 2
        // yk = x0 − 1 / (3L) * \delta3 - ((1 - tau) / (3L)) + tau * alpha)
        // * \delta2 - ((1-tau)^2 / (3L) + (1 - (1 - tau)^2) * alpha) * \delta1,
        // k = 3.
        y = iterate + 1.0 / (3.0 * lipschitz) * w;
      }
      else
      {
        y = iterate + tau1 * (zNew - z);
      }

      z = std::move(zNew);

      // sum_{j=0}^{m-1} 1 + std::min(alpha * convexity, 1 / (4 * m)^j * ys).
      w += cw * iterate;
      cw *= r;

      currentFunction += effectiveBatchSize;
      f += effectiveBatchSize;
    }
    iterate0 = normalizer * w;
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
