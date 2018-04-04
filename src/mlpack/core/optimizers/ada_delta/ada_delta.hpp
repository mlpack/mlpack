/**
 * @file ada_delta.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
 *
 * Implementation of the AdaDelta optimizer. AdaDelta is an optimizer that
 * dynamically adapts over time using only first order information.
 * Additionally, AdaDelta requires no manual tuning of a learning rate.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADA_DELTA_ADA_DELTA_HPP
#define MLPACK_CORE_OPTIMIZERS_ADA_DELTA_ADA_DELTA_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include "ada_delta_update.hpp"

namespace mlpack {
namespace optimization {

/**
 * AdaDelta is an optimizer that uses two ideas to improve upon the two main
 * drawbacks of the Adagrad method:
 *
 *  - Accumulate Over Window
 *  - Correct Units with Hessian Approximation
 *
 * For more information, see the following.
 *
 * @code
 * @article{Zeiler2012,
 *   author  = {Matthew D. Zeiler},
 *   title   = {{ADADELTA:} An Adaptive Learning Rate Method},
 *   journal = {CoRR},
 *   year    = {2012}
 * }
 * @endcode
 *
 * For AdaDelta to work, a DecomposableFunctionType template parameter is
 * required. This class must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates,
 *                   const size_t i,
 *                   const size_t batchSize);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient,
 *                 const size_t batchSize);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA (see mlpack::nca::NCA), NumFunctions() should return the number
 * of points in the dataset, and Evaluate(coordinates, 0) will evaluate the
 * objective function on the first point in the dataset (presumably, the dataset
 * is held internally in the DecomposableFunctionType).
 */
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
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in one step.
   * @param rho Smoothing constant.
   * @param epsilon Value used to initialise the mean squared gradient
   *        parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   */
  AdaDelta(const double stepSize = 1.0,
           const size_t batchSize = 32,
           const double rho = 0.95,
           const double epsilon = 1e-6,
           const size_t maxIterations = 100000,
           const double tolerance = 1e-5,
           const bool shuffle = true);

  /**
   * Optimize the given function using AdaDelta. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
   * objective value is returned. The DecomposableFunctionType is checked for
   * API consistency at compile time.
   *
   * @tparam DecomposableFunctionType Type of the function to optimize.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate)
  {
    return optimizer.Optimize(function, iterate);
  }

  //! Get the step size.
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the smoothing parameter.
  double Rho() const { return optimizer.UpdatePolicy().Rho(); }
  //! Modify the smoothing parameter.
  double& Rho() { return optimizer.UpdatePolicy().Rho(); }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return optimizer.UpdatePolicy().Epsilon(); }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return optimizer.UpdatePolicy().Epsilon(); }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return optimizer.MaxIterations(); }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return optimizer.MaxIterations(); }

  //! Get the tolerance for termination.
  double Tolerance() const { return optimizer.Tolerance(); }
  //! Modify the tolerance for termination.
  double& Tolerance() { return optimizer.Tolerance(); }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return optimizer.Shuffle(); }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return optimizer.Shuffle(); }

 private:
  //! The Stochastic Gradient Descent object with AdaDelta policy.
  SGD<AdaDeltaUpdate> optimizer;
};

} // namespace optimization
} // namespace mlpack

#endif
