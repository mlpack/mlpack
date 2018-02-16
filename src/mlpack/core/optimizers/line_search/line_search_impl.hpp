/**
 * @file line_search_impl.hpp
 * @author Chenzhe Diao
 *
 * Implementation of line search with secant method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_LINE_SEARCH_LINE_SEARCH_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_LINE_SEARCH_LINE_SEARCH_IMPL_HPP

// In case it hasn't been included yet.
#include "line_search.hpp"

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template<typename FunctionType>
double LineSearch::Optimize(FunctionType& function,
                            const arma::mat& x1,
                            arma::mat& x2)
{
  typedef Function<FunctionType> FullFunctionType;
  FullFunctionType& f = static_cast<FullFunctionType&>(function);

  // Check that we have all the functions we will need.
  traits::CheckFunctionTypeAPI<FullFunctionType>();

  // Set up the search line, that is,
  // find the zero of der(gamma) = Derivative(gamma).
  arma::mat deltaX = x2 - x1;
  double gamma = 0;
  double derivative = Derivative(f, x1, deltaX, 0);
  double derivativeNew = Derivative(f, x1, deltaX, 1);
  double secant = derivativeNew - derivative;

  if (derivative >= 0.0) // Optimal solution at left endpoint.
  {
    x2 = x1;
    return f.Evaluate(x1);
  }
  else if (derivativeNew <= 0.0) // Optimal solution at right endpoint.
  {
    return f.Evaluate(x2);
  }
  else if (secant < tolerance) // function too flat, just take left endpoint.
  {
    x2 = x1;
    return f.Evaluate(x1);
  }

  // Line search by Secant Method.
  for (size_t k = 0; k < maxIterations; ++k)
  {
    // secant should always >=0 for convex function.
    if (secant < 0.0)
    {
      Log::Fatal << "LineSearchSecant: Function is not convex!" << std::endl;
      x2 = x1;
      return function.Evaluate(x1);
    }

    // Solve new gamma.
    double gammaNew = gamma - derivative / secant;
    gammaNew = std::max(gammaNew, 0.0);
    gammaNew = std::min(gammaNew, 1.0);

    // Update secant, gamma and derivative
    derivativeNew = Derivative(function, x1, deltaX, gammaNew);
    secant = (derivativeNew - derivative) / (gammaNew - gamma);
    gamma = gammaNew;
    derivative = derivativeNew;

    if (std::fabs(derivative) < tolerance)
    {
      Log::Info << "LineSearchSecant: minimized within tolerance "
          << tolerance << "; " << "terminating optimization." << std::endl;
      x2 = (1 - gamma) * x1 + gamma * x2;
      return f.Evaluate(x2);
    }
  }

  Log::Info << "LineSearchSecant: maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;

  x2 = (1 - gamma) * x1 + gamma * x2;
  return f.Evaluate(x2);
}  // Optimize


//! Derivative of the function along the search line.
template<typename FunctionType>
double LineSearch::Derivative(FunctionType& function,
                              const arma::mat& x0,
                              const arma::mat& deltaX,
                              const double gamma)
{
  arma::mat gradient(x0.n_rows, x0.n_cols);
  function.Gradient(x0 + gamma * deltaX, gradient);
  return arma::dot(gradient, deltaX);
}


} // namespace optimization
} // namespace mlpack
#endif
