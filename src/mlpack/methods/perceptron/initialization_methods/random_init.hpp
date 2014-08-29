/**
 * @file random_init.hpp
 * @author Udit Saxena
 *
 * Random initialization for perceptron weights.
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
#ifndef _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP
#define _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {

/**
 * This class is used to initialize weights for the weightVectors matrix in a
 * random manner.
 */
class RandomInitialization
{
 public:
  RandomInitialization() { }

  inline static void Initialize(arma::mat& W,
                                const size_t row,
                                const size_t col)
  {
    W = arma::randu<arma::mat>(row, col);
  }
}; // class RandomInitialization

}; // namespace perceptron
}; // namespace mlpack

#endif
