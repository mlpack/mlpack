/**
 * @file test_function.cpp
 * @author Ryan Curtin
 *
 * Implementation of very simple test function for stochastic gradient descent
 * (SGD).
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "test_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

double SGDTestFunction::Evaluate(const arma::mat& coordinates, const size_t i)
    const
{
  switch (i)
  {
    case 0:
      return -std::exp(-std::abs(coordinates[0]));

    case 1:
      return std::pow(coordinates[1], 2);

    case 2:
      return std::pow(coordinates[2], 4) + 3 * std::pow(coordinates[2], 2);

    default:
      return 0;
  }
}

void SGDTestFunction::Gradient(const arma::mat& coordinates,
                               const size_t i,
                               arma::mat& gradient) const
{
  gradient.zeros(3);
  switch (i)
  {
    case 0:
      if (coordinates[0] >= 0)
        gradient[0] = std::exp(-coordinates[0]);
      else
        gradient[0] = -std::exp(coordinates[1]);
      break;

    case 1:
      gradient[1] = 2 * coordinates[1];
      break;

    case 2:
      gradient[2] = 4 * std::pow(coordinates[2], 3) + 6 * coordinates[2];
      break;
  }
}
