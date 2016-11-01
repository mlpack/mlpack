/**
 * @file gradient_descent_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Simple gradient descent implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP

// In case it hasn't been included yet.
#include "gradient_descent.hpp"

namespace mlpack {
namespace optimization {

template<typename FunctionType>
GradientDescent<FunctionType>::GradientDescent(
    FunctionType& function,
    const double stepSize,
    const size_t maxIterations,
    const double tolerance) :
    function(function),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename FunctionType>
double GradientDescent<FunctionType>::Optimize(
    arma::mat& iterate)
{
  // To keep track of where we are and how things are going.
  double overallObjective = function.Evaluate(iterate);
  double lastObjective = DBL_MAX;

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i != maxIterations; ++i)
  {
    // Output current objective function.
    Log::Info << "Gradient Descent: iteration " << i << ", objective " 
        << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Log::Warn << "Gradient Descent: converged to " << overallObjective 
          << "; terminating" << " with failure.  Try a smaller step size?" 
          << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Log::Info << "Gradient Descent: minimized within tolerance " 
          << tolerance << "; " << "terminating optimization." << std::endl;
      return overallObjective;
    }

    // Reset the counter variables.
    lastObjective = overallObjective;

    function.Gradient(iterate, gradient);

    // And update the iterate.
    iterate -= stepSize * gradient;

    // Now add that to the overall objective function.
    overallObjective = function.Evaluate(iterate);
  }

  Log::Info << "Gradient Descent: maximum iterations (" << maxIterations 
      << ") reached; " << "terminating optimization." << std::endl;
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
