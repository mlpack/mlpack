/**
 * @file zero_init.hpp
 * @author Udit Saxena
 *
 * Implementation of ZeroInitialization policy for perceptrons.
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
#ifndef _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP
#define _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {

/**
 * This class is used to initialize the matrix weightVectors to zero.
 */
class ZeroInitialization
{
 public:
  ZeroInitialization() { }

  inline static void Initialize(arma::mat& W,
                                const size_t row,
                                const size_t col)
  {
    arma::mat tempWeights(row, col);
    tempWeights.fill(0.0);

    W = tempWeights;
  }
}; // class ZeroInitialization

}; // namespace perceptron
}; // namespace mlpack

#endif
