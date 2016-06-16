/**
 * @file random_init.hpp
 * @author Udit Saxena
 *
 * Random initialization for perceptron weights.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP
#define MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP

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

  inline static void Initialize(arma::mat& weights,
                                arma::vec& biases,
                                const size_t numFeatures,
                                const size_t numClasses)
  {
    weights.randu(numFeatures, numClasses);
    biases.randu(numClasses);
  }
}; // class RandomInitialization

} // namespace perceptron
} // namespace mlpack

#endif
