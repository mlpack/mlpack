/**
 * @file colville_function.cpp
 * @author Marcus Edel
 *
 * Implementation of the Coville function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "colville_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

ColvilleFunction::ColvilleFunction() { /* Nothing to do here */ }

void ColvilleFunction::Shuffle() { /* Nothing to do here */ }

double ColvilleFunction::Evaluate(const arma::mat& coordinates,
                                  const size_t /* begin */,
                                  const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);
  const double x3 = coordinates(2);
  const double x4 = coordinates(3);

  const double objective = 100 * std::pow(std::pow(x1, 2) - x2, 2) +
      std::pow(x1 - 1, 2) + std::pow(x3 - 1, 2) + 90 *
      std::pow(std::pow(x3, 2) - x4, 2) + 10.1 * (std::pow(x2 - 1, 2) +
      std::pow(x4 - 1, 2)) + 19.8 * (x2 - 1) * (x4 - 1);

  return objective;
}

double ColvilleFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

void ColvilleFunction::Gradient(const arma::mat& coordinates,
                                const size_t /* begin */,
                                arma::mat& gradient,
                                const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);
  const double x3 = coordinates(2);
  const double x4 = coordinates(3);

  gradient.set_size(4, 1);
  gradient(0) = 2 * (200 * x1 * (std::pow(x1, 2) - x2) + x1 - 1);
  gradient(1) = 19.8 * x4 - 200 * std::pow(x1, 2) + 220.2 * x2 - 40;
  gradient(2) = 2 * (180 * x3 * (std::pow(x3, 2) - x4) + x3 - 1);
  gradient(3) = 200.2 * x4 + 19.8 * x2 - 180 * std::pow(x3, 2) - 40;
}

void ColvilleFunction::Gradient(const arma::mat& coordinates,
                                arma::mat& gradient) const
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}
