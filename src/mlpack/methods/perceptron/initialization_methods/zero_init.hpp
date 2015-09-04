/**
 * @file zero_init.hpp
 * @author Udit Saxena
 *
 * Implementation of ZeroInitialization policy for perceptrons.
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
                                const size_t numFeatures,
                                const size_t numClasses)
  {
    arma::mat tempWeights(numFeatures, numClasses);
    tempWeights.fill(0.0);

    W = tempWeights;
  }
}; // class ZeroInitialization

} // namespace perceptron
} // namespace mlpack

#endif
