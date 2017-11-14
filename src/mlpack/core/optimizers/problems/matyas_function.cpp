/**
 * @file matyas_function.cpp
 * @author Marcus Edel
 *
 * Implementation of the Matyas function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "matyas_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

MatyasFunction::MatyasFunction() { /* Nothing to do here */ }

void MatyasFunction::Shuffle() { /* Nothing to do here */ }

double MatyasFunction::Evaluate(const arma::mat& coordinates,
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

double MatyasFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

void MatyasFunction::Gradient(const arma::mat& coordinates,
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

void MatyasFunction::Gradient(const arma::mat& coordinates, arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}
