/**
 * @file random_init.hpp
 * @author Udit Saxena
 *
 * Random initialization for perceptron weights.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
