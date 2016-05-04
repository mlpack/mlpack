/**
 * @file sgdp.hpp
 * @author Ranjan Mondal
 *
 * Parallel Stochastic Gradient Descent (SGD).
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_PARALLELSGD_SGDP_HPP
#define __MLPACK_CORE_OPTIMIZERS_PARALLELSGD_SGDP_HPP

#include <mlpack/core.hpp>
#include<omp.h>
#include<vector>

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
class ParallelSGD
{
  public:
  /**
  * @param function Function to be optimized (minimized).
  * @param stepSize Step size for each iteration.
  * @param maxIterations Maximum number of iterations allowed (0 means no
  *     limit).
  * @param tolerance Maximum absolute tolerance to terminate algorithm.
  *
  **/
  ParallelSGD(DecomposableFunctionType& function,
      const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5);

  /**
   * Optimize the given function using  parallel stochastic gradient descent.
   *  The given starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
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

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }


 private:
  //! The instantiated function.
  DecomposableFunctionType& function;

  //! The step size for each example.
  double stepSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "sgdp_impl.hpp"

#endif
