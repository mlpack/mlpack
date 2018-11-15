/**
 * @file rastrigin_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Rastrigin function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_RASTRIGIN_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_RASTRIGIN_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "rastrigin_function.hpp"

namespace ens {
namespace test {

inline RastriginFunction::RastriginFunction(const size_t n) :
    n(n),
    visitationOrder(arma::linspace<arma::Row<size_t> >(0, n - 1, n))

{
  initialPoint.set_size(n, 1);
  initialPoint.fill(-3);
}

inline void RastriginFunction::Shuffle()
{
  visitationOrder = arma::shuffle(
      arma::linspace<arma::Row<size_t> >(0, n - 1, n));
}

inline double RastriginFunction::Evaluate(const arma::mat& coordinates,
                                          const size_t begin,
                                          const size_t batchSize) const
{
  double objective = 0.0;
  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    objective += std::pow(coordinates(p), 2) - 10.0 *
        std::cos(2.0 * arma::datum::pi * coordinates(p));
  }
  objective += 10.0 * n;

  return objective;
}

inline double RastriginFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void RastriginFunction::Gradient(const arma::mat& coordinates,
                                        const size_t begin,
                                        arma::mat& gradient,
                                        const size_t batchSize) const
{
  gradient.zeros(n, 1);

  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    gradient(p) += (10.0 * n) * (2 * (coordinates(p) + 10.0 * arma::datum::pi *
        std::sin(2.0 * arma::datum::pi * coordinates(p))));
  }
}

inline void RastriginFunction::Gradient(const arma::mat& coordinates,
                                        arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
