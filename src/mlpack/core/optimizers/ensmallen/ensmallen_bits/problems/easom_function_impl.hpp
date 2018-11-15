/**
 * @file easom_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Easom function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_EASOM_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_EASOM_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "easom_function.hpp"

namespace ens {
namespace test {

inline EasomFunction::EasomFunction() { /* Nothing to do here */ }

inline void EasomFunction::Shuffle() { /* Nothing to do here */ }

inline double EasomFunction::Evaluate(const arma::mat& coordinates,
                                      const size_t /* begin */,
                                      const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = -std::cos(x1) * std::cos(x2) *
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2));

  return objective;
}

inline double EasomFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void EasomFunction::Gradient(const arma::mat& coordinates,
                                    const size_t /* begin */,
                                    arma::mat& gradient,
                                    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = 2 * (x1 - arma::datum::pi) *
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2)) *
      std::cos(x1) * std::cos(x2) +
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 -  arma::datum::pi, 2)) *
      std::sin(x1) * std::cos(x2);

  gradient(1) = 2 * (x2 - arma::datum::pi) *
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2)) *
      std::cos(x1) * std::cos(x2) +
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2)) *
      std::cos(x1) * std::sin(x2);
}

inline void EasomFunction::Gradient(const arma::mat& coordinates,
                                    arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
