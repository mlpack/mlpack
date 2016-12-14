/**
 * @file rmsprop_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the RMSprop optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_RMSPROP_RMSPROP_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_RMSPROP_RMSPROP_IMPL_HPP

// In case it hasn't been included yet.
#include "rmsprop.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
RMSprop<DecomposableFunctionType>::RMSprop(DecomposableFunctionType& function,
                                           const double stepSize,
                                           const double alpha,
                                           const double eps,
                                           const size_t maxIterations,
                                           const double tolerance,
                                           const bool shuffle) :
    function(function),
    stepSize(stepSize),
    alpha(alpha),
    eps(eps),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double RMSprop<DecomposableFunctionType>::Optimize(arma::mat& iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // This is used only if shuffle is true.
  arma::Col<size_t> visitationOrder;
  if (shuffle)
    visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
        (numFunctions - 1), numFunctions));

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);

  // Leaky sum of squares of parameter gradient.
  arma::mat meanSquaredGradient = arma::zeros<arma::mat>(iterate.n_rows,
      iterate.n_cols);

  for (size_t i = 1; i != maxIterations; ++i, ++currentFunction)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      // Output current objective function.
      Log::Info << "RMSprop: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "RMSprop: converged to " << overallObjective
            << "; terminating with failure. Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "RMSprop: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        visitationOrder = arma::shuffle(visitationOrder);
    }

    // Evaluate the gradient for this iteration.
    if (shuffle)
      function.Gradient(iterate, visitationOrder[currentFunction], gradient);
    else
      function.Gradient(iterate, currentFunction, gradient);

    // And update the iterate.
    meanSquaredGradient *= alpha;
    meanSquaredGradient += (1 - alpha) * (gradient % gradient);
    iterate -= stepSize * gradient / (arma::sqrt(meanSquaredGradient) + eps);

    // Now add that to the overall objective function.
    if (shuffle)
      overallObjective += function.Evaluate(iterate,
          visitationOrder[currentFunction]);
    else
      overallObjective += function.Evaluate(iterate, currentFunction);
  }

  Log::Info << "RMSprop: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;
  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
