/**
 * @file drop_wave_function.cpp
 * @author Marcus Edel
 *
 * Implementation of the Drop-Wave function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "drop_wave_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

DropWaveFunction::DropWaveFunction() { /* Nothing to do here */ }

void DropWaveFunction::Shuffle() { /* Nothing to do here */ }

double DropWaveFunction::Evaluate(const arma::mat& coordinates,
                                  const size_t /* begin */,
                                  const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = -1.0 * (1.0 + std::cos(12.0 *
      std::sqrt(std::pow(x1, 2) + std::pow(x2, 2)))) /
      (0.5 * (std::pow(x1, 2) + std::pow(x2, 2)) + 2.0);

  return objective;
}

double DropWaveFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, 1);
}

void DropWaveFunction::Gradient(const arma::mat& coordinates,
                                const size_t /* begin */,
                                arma::mat& gradient,
                                const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = (12.0 * x1 * std::sin(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2)))) / (std::sqrt(std::pow(x1, 2) + std::pow(x2, 2)) *
      (0.5 * (std::pow(x1, 2) + std::pow(x2, 2)) + 2)) -
      (x1 * (-1.0 * std::cos(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2))) -1.0)) / std::pow(0.5 *
      (std::pow(x1, 2) + std::pow(x2, 2)) + 2, 2);

  gradient(1) = (12.0 * x2 * std::sin(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2)))) / (std::sqrt(std::pow(x1, 2) + std::pow(x2, 2)) *
      (0.5 * (std::pow(x1, 2) + std::pow(x2, 2)) + 2)) -
      (x2 * (-1.0 * std::cos(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2))) -1.0)) / std::pow(0.5 *
      (std::pow(x1, 2) + std::pow(x2, 2)) + 2, 2);
}

void DropWaveFunction::Gradient(const arma::mat& coordinates,
                                arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}
