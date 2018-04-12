/**
 * @file rosenbrock_function.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Rosenbrock function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "rosenbrock_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

RosenbrockFunction::RosenbrockFunction() { /* Nothing to do here */ }

void RosenbrockFunction::Shuffle() { /* Nothing to do here */ }

double RosenbrockFunction::Evaluate(const arma::mat& coordinates,
                                    const size_t /* begin */,
                                    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2), 2) +
                           /* f2(x) */ std::pow(1 - x1, 2);

  return objective;
}

double RosenbrockFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

void RosenbrockFunction::Gradient(const arma::mat& coordinates,
                                  const size_t /* begin */,
                                  arma::mat& gradient,
                                  const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = -2 * (1 - x1) + 400 * (std::pow(x1, 3) - x2 * x1);
  gradient(1) = 200 * (x2 - std::pow(x1, 2));
}

void RosenbrockFunction::Gradient(const arma::mat& coordinates,
                                  arma::mat& gradient) const
{
  Gradient(coordinates, 0, gradient, 1);
}
