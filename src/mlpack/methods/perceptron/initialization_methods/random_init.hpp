/**
 * @file methods/perceptron/initialization_methods/random_init.hpp
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

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class is used to initialize weights for the weightVectors matrix in a
 * random manner.
 */
class RandomPerceptronInitialization
{
 public:
  RandomPerceptronInitialization() { }

  template<typename eT>
  inline static void Initialize(arma::Mat<eT>& weights,
                                arma::Col<eT>& biases,
                                const size_t numFeatures,
                                const size_t numClasses)
  {
    weights.randu(numFeatures, numClasses);
    biases.randu(numClasses);
  }
}; // class RandomPerceptronInitialization

} // namespace mlpack

#endif
