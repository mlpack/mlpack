/**
 * @file random_init.hpp
 * @author Udit Saxena
 *
 * Random initialization for perceptron weights.
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
                                const size_t numFeatures,
                                const size_t numClasses)
  {
    W = arma::randu<arma::mat>(numFeatures, numClasses);
  }
}; // class RandomInitialization

} // namespace perceptron
} // namespace mlpack

#endif
