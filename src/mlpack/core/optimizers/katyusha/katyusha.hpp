/**
 * @file katyusha.hpp
 * @author Marcus Edel
 *
 * Katyusha a direct, primal-only stochastic gradient method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_KATYUSHA_KATYUSHA_HPP
#define MLPACK_CORE_OPTIMIZERS_KATYUSHA_KATYUSHA_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Katyusha is a direct, primal-only stochastic gradient method which uses a
 * negative momentum” on top of Nesterov’s momentum.
 *
 * For more information, see the following.
 *
 * @code
 * @article{2016arXiv160305953A,
 *   author  = {{Allen-Zhu}, Z.},
 *   title   = {Katyusha: The First Direct Acceleration of Stochastic Gradient
 *              Methods},
 *   journal = {ArXiv e-prints},
 *   url     = {https://arxiv.org/abs/1603.05953}
 *   year    = 2016,
 * }
 * @endcode
 *
 * For Katyusha to work, a DecomposableFunctionType template parameter
 * is required. This class must implement the following function:
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
class Katyusha
{
 public:
  /**
   * Construct the Katyusha optimizer with the given function and parameters.
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param lambda The regularization parameter.
   * @param tau The momentum parameter.
   * @param batchSize Batch size to use for each step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   */
  Katyusha(const double stepSize = 1000,
           const double lambda = 0.01,
           const double tau = 0.5,
           const size_t batchSize = 32,
           const size_t maxIterations = 100000,
           const double tolerance = 1e-5,
           const bool shuffle = true);

  /**
   * Optimize the given function using Katyusha. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the regularization parameter.
  double Lambda() const { return lambda; }
  //! Modify the regularization parameter.
  double& Lambda() { return lambda; }

  //! Get the regularization parameter.
  double Tau() const { return tau; }
  //! Modify the regularization parameter.
  double& Tau() { return tau; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

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
  //! The step size for each example.
  double stepSize;

  //! Regularization parameter.
  double lambda;

  //! The momentum parameter.
  double tau;

  //! The batch size for processing.
  size_t batchSize;

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
#include "katyusha_impl.hpp"

#endif
