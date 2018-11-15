/**
 * @file matyas_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Matyas function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_MATYAS_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_MATYAS_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "matyas_function.hpp"

namespace ens {
namespace test {

inline MatyasFunction::MatyasFunction() { /* Nothing to do here */ }

inline void MatyasFunction::Shuffle() { /* Nothing to do here */ }

inline double MatyasFunction::Evaluate(const arma::mat& coordinates,
                                       const size_t /* begin */,
                                       const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = 0.26 * (pow(x1, 2) + std::pow(x2, 2)) -
    0.48 * x1 * x2;

  return objective;
}

inline double MatyasFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void MatyasFunction::Gradient(const arma::mat& coordinates,
                                     const size_t /* begin */,
                                     arma::mat& gradient,
                                     const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = 0.52 * x1 - 48 * x2;
  gradient(1) = 0.52 * x2 - 0.48 * x1;
}

inline void MatyasFunction::Gradient(const arma::mat& coordinates,
                                     arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
