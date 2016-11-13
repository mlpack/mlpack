/**
 * @file gradient_descent.hpp
 * @author Sumedh Ghaisas
 *
 * Simple Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
#define MLPACK_CORE_OPTIMIZERS_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * Gradient Descent is a technique to minimize a function. To find a local 
 * minimum of a function using gradient descent, one takes steps proportional 
 * to the negative of the gradient of the function at the current point, 
 * producing the following update scheme:
 *
 * \f[
 * A_{j + 1} = A_j + \alpha \nabla F(A)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size. \f$ F \f$ 
 * is the function being optimized. The algorithm continues until \f$ j
 * \f$ reaches the maximum number of iterations---or when an update produces 
 * an improvement within a certain tolerance \f$ \epsilon \f$.  That is,
 *
 * \f[
 * | F(A_{j + 1}) - F(A_j) | < \epsilon.
 * \f]
 *
 * The parameter \f$\epsilon\f$ is specified by the tolerance parameter to the
 * constructor.
 *
 * For Gradient Descent to work, a FunctionType template parameter is required.
 * This class must implement the following function:
 *
 *   double Evaluate(const arma::mat& coordinates);
 *   void Gradient(const arma::mat& coordinates,
 *                 arma::mat& gradient);
 *
 * @tparam FunctionType Decomposable objective function type to be
 *     minimized.
 */
template<typename FunctionType>
class GradientDescent
{
 public:
  /**
   * Construct the Gradient Descent optimizer with the given function and 
   * parameters.  The defaults here are not necessarily good for the given 
   * problem, so it is suggested that the values used be tailored to the task 
   * at hand.
   *
   * @param function Function to be optimized (minimized).
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   */
  GradientDescent(FunctionType& function,
      const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5);

  /**
   * Optimize the given function using gradient descent.  The given starting 
   * point will be modified to store the finishing point of the algorithm, and 
   * the final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate);

  //! Get the instantiated function to be optimized.
  const FunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  FunctionType& Function() { return function; }

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
  FunctionType& function;

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
#include "gradient_descent_impl.hpp"

#endif
