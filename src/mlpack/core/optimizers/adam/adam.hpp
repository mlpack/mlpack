/**
 * @file adam.hpp
 * @author Marcus Edel
 *
 * Implementation of the Adam optimizer. Adam is an an algorithm for first-
 * order gradient-based optimization of stochastic objective functions, based on
 * adaptive estimates of lower-order moments.
 */
#ifndef __MLPACK_METHODS_ANN_OPTIMIZER_ADAM_HPP
#define __MLPACK_METHODS_ANN_OPTIMIZER_ADAM_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * Adam is an optimizer that computes individual adaptive learning rates for
 * different parameters from estimates of first and second moments of the
 * gradients.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Kingma2014,
 *   author    = {Diederik P. Kingma and Jimmy Ba},
 *   title     = {Adam: {A} Method for Stochastic Optimization},
 *   journal   = {CoRR},
 *   year      = {2014}
 * }
 * @endcode
 */
template<typename DecomposableFunctionType>
class Adam
{
 public:
  /**
   * Construct the Adam optimizer with the given function and parameters.
   *
   * @param function Function to be optimized (minimized).
   * @param lr The learning rate coefficient.
   * @param beta1 The first moment coefficient.
   * @param beta2 The second moment coefficient.
   * @param eps The eps coefficient to avoid division by zero (numerical
   *        stability).
   */
  Adam(DecomposableFunctionType& function,
          const double stepSize = 0.01,
	  const double beta1 = 0.9,
          const double beta2 = 0.999,
          const double eps = 1e-8,
	  const size_t maxIterations = 100000,
      	  const double tolerance = 1e-5,
          const bool shuffle = true);

   /**
   * Optimize the given function using Adam. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate);
  
  //! Get the instantiated function to be optimized.
  const DecomposableFunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  DecomposableFunctionType& Function() { return function; }

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the first moment coefficient.
  double Beta1() const { return beta1; }
  //! Modify the first moment coefficient.
  double& Beta1() { return beta1; }

  //! Get the second moment coefficient.
  double Beta2() const { return beta2; }
  //! Modify the second moment coefficient.
  double& Beta2() { return beta2; }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return eps; }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return eps; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

 private:
  //! The instantiated function.
  DecomposableFunctionType& function;

  //! The step size for each example.
  double stepSize;

  //! The first moment coefficient.
  const double beta1;

  //! The second moment coefficient.
  const double beta2;

  //! The value used to initialise the mean squared gradient parameter.
  double eps;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "adam_impl.hpp"

#endif
