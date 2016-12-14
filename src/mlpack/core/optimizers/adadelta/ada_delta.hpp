/**
 * @file ada_delta.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 *
 * Implementation of the Adadelta optimizer. Adadelta is an optimizer that
 * dynamically adapts over time using only first order information.
 * Additionally, Adadelta requires no manual tuning of a learning rate.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_ADADELTA_ADA_DELTA_HPP
#define __MLPACK_CORE_OPTIMIZERS_ADADELTA_ADA_DELTA_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * Adadelta is an optimizer that uses two ideas to improve upon the two main
 * drawbacks of the Adagrad method:
 *
 *  - Accumulate Over Window
 *  - Correct Units with Hessian Approximation
 *
 * For more information, see the following.
 *
 * @code
 * @article{Zeiler2012,
 *   author    = {Matthew D. Zeiler},
 *   title     = {{ADADELTA:} An Adaptive Learning Rate Method},
 *   journal   = {CoRR},
 *   year      = {2012}
 * }
 * @endcode
 *

 * For AdaDelta to work, a DecomposableFunctionType template parameter is
 * required. This class must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates, const size_t i);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA (see mlpack::nca::NCA), NumFunctions() should return the number
 * of points in the dataset, and Evaluate(coordinates, 0) will evaluate the
 * objective function on the first point in the dataset (presumably, the dataset
 * is held internally in the DecomposableFunctionType).
 *
 * @tparam DecomposableFunctionType Decomposable objective function type to be
 *     minimized.
 */
template<typename DecomposableFunctionType>
class AdaDelta
{
 public:
  /**
   * Construct the AdaDelta optimizer with the given function and parameters.
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand. The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param function Function to be optimized (minimized).
   * @param rho Smoothing constant
   * @param eps Value used to initialise the mean squared gradient parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   */
  AdaDelta(DecomposableFunctionType& function,
      const double rho = 0.95,
      const double eps = 1e-6,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const bool shuffle = true);

  /**
   * Optimize the given function using AdaDelta. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
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

  //! Get the smoothing parameter.
  double Rho() const { return rho; }
  //! Modify the smoothing parameter.
  double& Rho() { return rho; }

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

  //! The smoothing parameter.
  double rho;

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
#include "ada_delta_impl.hpp"

#endif

