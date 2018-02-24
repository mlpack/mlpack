/**
 * @file sarah.hpp
 * @author Marcus Edel
 *
 * StochAstic Recusive gRadient algoritHm (SARAH).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SARAH_SARAH_HPP
#define MLPACK_CORE_OPTIMIZERS_SARAH_SARAH_HPP

#include <mlpack/prereqs.hpp>

#include "sarah_update.hpp"
#include "sarah_plus_update.hpp"

namespace mlpack {
namespace optimization {

/**
 * StochAstic Recusive gRadient algoritHm (SARAH). is a variance reducing
 * stochastic recursive gradient algorithm for minimizing a function
 * which can be expressed as a sum of other functions.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Nguyen2017,
 *   author  = {{Nguyen}, L.~M. and {Liu}, J. and {Scheinberg},
 *              K. and {Tak{\'a}{\v c}}, M.},
 *   title   = {SARAH: A Novel Method for Machine Learning Problems Using
 *              Stochastic Recursive Gradient},
 *   journal = {ArXiv e-prints},
 *   url     = {https://arxiv.org/abs/1703.00102}
 *   year    = 2017,
 * }
 * @endcode
 *
 * For SARAH to work, a DecomposableFunctionType template parameter is required.
 * This class must implement the following function:
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
 * objective function on the first point in the dataset (
 * is held internally in the DecomposableFunctionType).
 *
 * @tparam UpdatePolicyType update policy used by SARAHType during the iterative
 *    update process.
 */
template<typename UpdatePolicyType = SARAHUpdate>
class SARAHType
{
 public:
  /**
   * Construct the SARAH optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Batch size to use for each step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param innerIterations The number of inner iterations allowed (0 means
   *    n / batchSize). Note that the full gradient is only calculated in
   *    the outer iteration.
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   * @param updatePolicy Instantiated update policy used to adjust the given
   *     parameters.
   */
  SARAHType(const double stepSize = 0.01,
            const size_t batchSize = 32,
            const size_t maxIterations = 1000,
            const size_t innerIterations = 0,
            const double tolerance = 1e-5,
            const bool shuffle = true,
            const UpdatePolicyType& updatePolicy = UpdatePolicyType());

  /**
   * Optimize the given function using SARAH. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
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

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the maximum number of iterations (0 indicates default n / b).
  size_t InnerIterations() const { return innerIterations; }
  //! Modify the maximum number of iterations (0 indicates default n / b).
  size_t& InnerIterations() { return innerIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

  //! Get the update policy.
  const UpdatePolicyType& UpdatePolicy() const { return updatePolicy; }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy() { return updatePolicy; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The batch size for processing.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The maximum number of allowed inner iterations per epoch.
  size_t innerIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;

  //! The update policy used to update the parameters in each iteration.
  UpdatePolicyType updatePolicy;
};

// Convenience typedefs.

/**
 * Standard stochastic variance reduced gradient.
 */
using SARAH = SARAHType<SARAHUpdate>;

/**
 * Stochastic variance reduced gradient with Barzilai-Borwein.
 */
using SARAH_Plus = SARAHType<SARAHPlusUpdate>;

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "sarah_impl.hpp"

#endif
