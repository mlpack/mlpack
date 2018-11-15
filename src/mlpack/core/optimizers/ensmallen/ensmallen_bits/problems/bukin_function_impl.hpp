/**
 * @file bukin_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Bukin function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_BUKIN_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_BUKIN_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "bukin_function.hpp"

namespace ens {
namespace test {

inline BukinFunction::BukinFunction(const double epsilon) : epsilon(epsilon)
{ /* Nothing to do here */ }

inline void BukinFunction::Shuffle() { /* Nothing to do here */ }

inline double BukinFunction::Evaluate(const arma::mat& coordinates,
                                      const size_t /* begin */,
                                      const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = 100 * std::sqrt(std::abs(x2 - 0.01 *
      std::pow(x1, 2))) + 0.01 * std::abs(x1 + 10);

  return objective;
}

inline double BukinFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void BukinFunction::Gradient(const arma::mat& coordinates,
                                    const size_t /* begin */,
                                    arma::mat& gradient,
                                    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = (0.01 * (x1 + 10.0)) / (std::abs(x1 + 10.0) + epsilon) -
      (x1 * (x2 - 0.01 * std::pow(x1, 2))) / std::pow(std::abs(x2 - 0.01 *
      std::pow(x1, 2)), 1.5);
  gradient(1) = (50 * (x2 - 0.01 * std::pow(x1, 2))) /
      std::pow(std::abs(x2 - 0.01 * std::pow(x1, 2)), 1.5);
}

inline void BukinFunction::Gradient(const arma::mat& coordinates,
                                    arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
