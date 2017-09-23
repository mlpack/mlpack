/**
 * @file iqn_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of an incremental Quasi-Newton with local superlinear
 * convergence rate as proposed by A. Mokhtari et al. in "IQN: An Incremental
 * Quasi-Newton Method with Local Superlinear Convergence Rate".
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_IQN_IQN_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_IQN_IQN_IMPL_HPP

// In case it hasn't been included yet.
#include "iqn.hpp"

namespace mlpack {
namespace optimization {

IQN::IQN(const double stepSize,
         const size_t maxIterations,
         const double tolerance) :
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double IQN::Optimize(DecomposableFunctionType& function, arma::mat& iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  double overallObjective = 0;

  arma::cube y(iterate.n_rows, iterate.n_cols, numFunctions);
  arma::cube t(iterate.n_rows, iterate.n_cols, numFunctions);
  arma::cube Q(iterate.n_elem, iterate.n_elem, numFunctions);

  t.slice(0) = arma::randn(iterate.n_rows, iterate.n_cols);

  arma::mat g = arma::zeros(iterate.n_rows, iterate.n_cols);
  for (size_t i = 0; i < numFunctions; ++i)
  {
    if (i > 0)
      t.slice(i) = t.slice(0);

    function.Gradient(t.slice(0), i, y.slice(i));
    Q.slice(i).eye();

    g += y.slice(i);
  }
  g /= numFunctions;

  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat B = arma::eye(iterate.n_elem, iterate.n_elem);
  arma::mat u = t.slice(0);

  for (size_t i = 1; i != maxIterations; ++i)
  {
    for (size_t j = 0; j < numFunctions; ++j)
    {
      // Cyclicly iterating through the nnumber of functions.
      const size_t it = ((j + 1) % numFunctions);

      if (arma::norm(iterate - t.slice(it)) > 0)
      {
        function.Gradient(iterate, it, gradient);
        const arma::mat s = iterate - t.slice(it);
        const arma::mat yy = gradient - y.slice(it);

        if (iterate.n_rows > iterate.n_cols)
        {
          const arma::mat stochasticHessian = Q.slice(it) + yy * yy.t() /
              arma::as_scalar(yy.t() * s) - Q.slice(it) * s * s.t() *
              Q.slice(it) / arma::as_scalar(s.t() * Q.slice(it) * s);

           // Update aggregate Hessian approximation.
           B += (1.0 / numFunctions) * (stochasticHessian - Q.slice(it));

           // Update aggregate Hessian-variable product.
           u += (1.0 / numFunctions) * (stochasticHessian * iterate -
              Q.slice(it) * t.slice(it));

           // Update aggregate gradient.
           g += (1.0 / numFunctions) * (gradient - y.slice(it));

           // Update the function information tables.
           Q.slice(it) = stochasticHessian;
           y.slice(it) = gradient;
           t.slice(it) = iterate;

           iterate = stepSize * B.i() * (u - g) + (1 - stepSize) * iterate;
        }
        else
        {
          const arma::mat stochasticHessian = Q.slice(it).t() + yy.t() * yy /
              arma::as_scalar(yy * s.t()) - Q.slice(it).t() * s.t() * s *
              Q.slice(it).t() / arma::as_scalar(s * Q.slice(it).t() * s.t());

          // Update aggregate Hessian approximation.
          B += (1.0 / numFunctions) * (stochasticHessian - Q.slice(it));

          // Update aggregate Hessian-variable product.
          u += (1.0 / numFunctions) * (iterate * stochasticHessian -
              t.slice(it) * Q.slice(it));

          // Update aggregate gradient.
          g += (1.0 / numFunctions) * (gradient - y.slice(it));

          // Update the function information tables.
          Q.slice(it) = stochasticHessian;
          y.slice(it) = gradient;
          t.slice(it) = iterate;

          iterate = stepSize * (u - g) * B.i() + (1 - stepSize) * iterate;
        }
      }
    }

    overallObjective = 0;
    for (size_t i = 0; i < numFunctions; ++i)
      overallObjective += function.Evaluate(iterate, i);
    overallObjective /= numFunctions;

    // Output current objective function.
    Log::Info << "IQN: iteration " << i << ", objective " << overallObjective
        << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Log::Warn << "IQN: converged to " << overallObjective << "; terminating"
          << " with failure.  Try a smaller step size?" << std::endl;
      return overallObjective;
    }

    if (overallObjective < tolerance)
    {
      Log::Info << "IQN: minimized within tolerance " << tolerance << "; "
          << "terminating optimization." << std::endl;
      return overallObjective;
    }
  }

  Log::Info << "IQN: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
