/**
 * @file zero_init.hpp
 * @author Udit Saxena
 *
 * Implementation of ZeroInitialization policy for perceptrons.
 */
#ifndef __MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP
#define __MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP

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

  inline static void Initialize(arma::mat& weights,
                                arma::vec& biases,
                                const size_t numFeatures,
                                const size_t numClasses)
  {
    weights.zeros(numFeatures, numClasses);
    biases.zeros(numClasses);
  }
}; // class ZeroInitialization

} // namespace perceptron
} // namespace mlpack

#endif
