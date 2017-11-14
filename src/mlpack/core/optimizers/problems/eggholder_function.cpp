/**
 * @file eggholder_function.cpp
 * @author Marcus Edel
 *
 * Implementation of the Eggholder function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "eggholder_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

EggholderFunction::EggholderFunction() { /* Nothing to do here */ }

void EggholderFunction::Shuffle() { /* Nothing to do here */ }

double EggholderFunction::Evaluate(const arma::mat& coordinates,
                                   const size_t /* begin */,
                                   const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = -1.0 * (x2 + 47) * std::sin(std::sqrt(
      std::abs(x2 + x1 / 2 + 47))) - x1 * std::sin(std::sqrt(
      std::abs(x1 - (x2 + 47))));

  return objective;
}

double EggholderFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

void EggholderFunction::Gradient(const arma::mat& coordinates,
                                 const size_t /* begin */,
                                 arma::mat& gradient,
                                 const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = -1.0 * std::sin(std::sqrt(std::abs(x1 - x2 - 47))) -
      (x1 * (x1 - x2 - 47) * std::cos(std::sqrt(std::abs(x1 - x2 - 47)))) /
      std::pow(2 * std::abs(x1 - x2 - 47), 1.5) -
      ((x1 + 47) * (x1 / 2 + x2 + 47) *
      std::cos(std::sqrt(std::abs(x1 / 2 + x2  + 47)))) /
      (4 * std::pow(std::abs(x1 / 2 + x2  + 47), 1.5));

  gradient(1) = -1.0 * std::sin(std::sqrt(std::abs(x1 / 2 + x2 + 47))) -
      (x1 * (x1 - x2 - 47) * std::cos(std::sqrt(std::abs(x1 - x2 - 47)))) /
      std::pow(2 * std::abs(x1 - x2 - 47), 1.5) -
      ((x1 + 47) * (x1 / 2 + x2 + 47) *
      std::cos(std::sqrt(std::abs(x1 / 2 + x2  + 47)))) /
      (4 * std::pow(std::abs(x1 / 2 + x2  + 47), 1.5));
}

void EggholderFunction::Gradient(const arma::mat& coordinates,
                                 arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}
