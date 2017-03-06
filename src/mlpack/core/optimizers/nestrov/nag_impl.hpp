/**
 * @file nag_impl.hpp
 * @author Ryan Curtin
 * @author Kris Singh
 *
 * Implementation of stochastic gradient descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_NAG_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_NAG__IMPL_HPP

#include <mlpack/methods/regularized_svd/regularized_svd_function.hpp>
// In case it hasn't been included yet.
#include "nag.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
NAG<DecomposableFunctionType>::NAG(DecomposableFunctionType& function,
                                   const double stepSize,
                                   const size_t maxIterations,
                                   const double tolerance,
                                   const double momentum,
                                   const bool shuffle) :
    function(function),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    momentum(momentum),
    shuffle(shuffle)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double NAG<DecomposableFunctionType>::Optimize(arma::mat& iterate)
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
  arma::mat velocity = arma::zeros<arma::mat>(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i != maxIterations; ++i, ++currentFunction)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      // Output current objective function.
      Log::Info << "NAG: iteration " << i << ", objective " << overallObjective
          << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "NAG: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "NAG: minimized within tolerance " << tolerance << "; "
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

    // Take step in the momentum direction in parameter space find gradient
    iterate -= momentum * velocity;

    // Evaluate the gradient at the modified iteration
    if (shuffle)
      function.Gradient(iterate, visitationOrder[currentFunction], gradient);
    else
      function.Gradient(iterate, currentFunction, gradient);

    // Restore iterate back
    iterate += momentum * velocity;

    //Compute velocity update
    velocity = velocity*momentum + stepSize*gradient;
    
    // And update the iterate.
    iterate += velocity;

    // Now add that to the overall objective function.
    if (shuffle)
      overallObjective += function.Evaluate(iterate,
          visitationOrder[currentFunction]);
    else
      overallObjective += function.Evaluate(iterate, currentFunction);
  }

  Log::Info << "NAG: maximum iterations (" << maxIterations << ") reached; "
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
