/**
 * @file wood_function.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Wood function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "wood_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

WoodFunction::WoodFunction() { /* Nothing to do here */ }

void WoodFunction::Shuffle() { /* Nothing to do here */ }

double WoodFunction::Evaluate(const arma::mat& coordinates,
                              const size_t /* begin */,
                              const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);
  const double x3 = coordinates(2);
  const double x4 = coordinates(3);

  const double objective = /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2), 2) +
                           /* f2(x) */ std::pow(1 - x1, 2) +
                           /* f3(x) */ 90 * std::pow(x4 - std::pow(x3, 2), 2) +
                           /* f4(x) */ std::pow(1 - x3, 2) +
                           /* f5(x) */ 10 * std::pow(x2 + x4 - 2, 2) +
                           /* f6(x) */ (1.0 / 10.0) * std::pow(x2 - x4, 2);

  return objective;
}

double WoodFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

void WoodFunction::Gradient(const arma::mat& coordinates,
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
  gradient(0) = 400 * (std::pow(x1, 3) - x2 * x1) - 2 * (1 - x1);
  gradient(1) = 200 * (x2 - std::pow(x1, 2)) + 20 * (x2 + x4 - 2) +
      (1.0 / 5.0) * (x2 - x4);
  gradient(2) = 360 * (std::pow(x3, 3) - x4 * x3) - 2 * (1 - x3);
  gradient(3) = 180 * (x4 - std::pow(x3, 2)) + 20 * (x2 + x4 - 2) -
      (1.0 / 5.0) * (x2 - x4);
}

void WoodFunction::Gradient(const arma::mat& coordinates, arma::mat& gradient)
    const
{
  Gradient(coordinates, 0, gradient, 1);
}
