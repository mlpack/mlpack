/**
 * @file rastrigin_function.cpp
 * @author Marcus Edel
 *
 * Implementation of the Rastrigin function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "rastrigin_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

RastriginFunction::RastriginFunction(const size_t n) :
    n(n),
    visitationOrder(arma::linspace<arma::Row<size_t> >(0, n - 1, n))

{
  initialPoint.set_size(n, 1);
  initialPoint.fill(-3);
}

void RastriginFunction::Shuffle()
{
  visitationOrder = arma::shuffle(
      arma::linspace<arma::Row<size_t> >(0, n - 1, n));
}

double RastriginFunction::Evaluate(const arma::mat& coordinates,
                                   const size_t begin,
                                   const size_t batchSize) const
{
  double objective = 0.0;
  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    objective += std::pow(coordinates(p), 2) - 10.0 *
        std::cos(2.0 * M_PI * coordinates(p));
  }
  objective += 10.0 * n;

  return objective;
}

double RastriginFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

void RastriginFunction::Gradient(const arma::mat& coordinates,
                                 const size_t begin,
                                 arma::mat& gradient,
                                 const size_t batchSize) const
{
  gradient.zeros(n, 1);

  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    gradient(p) += (10.0 * n) * (2 * (coordinates(p) + 10.0 * M_PI *
        std::sin(2.0 * M_PI * coordinates(p))));
  }
}

void RastriginFunction::Gradient(const arma::mat& coordinates,
                                 arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}
