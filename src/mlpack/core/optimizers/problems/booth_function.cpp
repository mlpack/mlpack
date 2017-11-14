/**
 * @file booth_function.cpp
 * @author Marcus Edel
 *
 * Implementation of the Booth function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "booth_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BoothFunction::BoothFunction() { /* Nothing to do here */ }

void BoothFunction::Shuffle() { /* Nothing to do here */ }

double BoothFunction::Evaluate(const arma::mat& coordinates,
                               const size_t /* begin */,
                               const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = std::pow(x1 + 2 * x2 - 7, 2) +
      std::pow(2 * x1 + x2 - 5, 2);

  return objective;
}

double BoothFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

void BoothFunction::Gradient(const arma::mat& coordinates,
                             const size_t /* begin */,
                             arma::mat& gradient,
                             const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = 10 * x1 + 8 * x2 - 34;
  gradient(1) = 8 * x1 + 10 * x2 - 38;
}

void BoothFunction::Gradient(const arma::mat& coordinates, arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}
