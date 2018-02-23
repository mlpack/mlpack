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
 * "negative momentum" on top of Nesterov’s momentum.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Allen-Zhu2016,
 *   author    = {{Allen-Zhu}, Z.},
 *   title     = {Katyusha: The First Direct Acceleration of Stochastic Gradient
 *                Methods},
 *   booktitle = {Proceedings of the 49th Annual ACM SIGACT Symposium on Theory
 *                of Computing},
 *   pages     = {1200--1205},
 *   publisher = {ACM},
 *   year      = {2017},
 *   series    = {STOC 2017},
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
 *
 * @tparam proximal Whether the proximal update should be used or not.
 */
template<bool Proximal = false>
class KatyushaType
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
   * @param convexity The regularization parameter.
   * @param lipschitz The Lipschitz constant.
   * @param batchSize Batch size to use for each step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *    limit).
   * @param innerIterations The number of inner iterations allowed (0 means
   *    n / batchSize). Note that the full gradient is only calculated in
   *    the outer iteration.
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *    function is visited in linear order.
   */
  KatyushaType(const double convexity = 1.0,
               const double lipschitz = 10.0,
               const size_t batchSize = 32,
               const size_t maxIterations = 1000,
               const size_t innerIterations = 0,
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

  //! Get the convexity parameter.
  double Convexity() const { return convexity; }
  //! Modify the convexity parameter.
  double& Convexity() { return convexity; }

  //! Get the lipschitz parameter.
  double Lipschitz() const { return lipschitz; }
  //! Modify the lipschitz parameter.
  double& Lipschitz() { return lipschitz; }

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

 private:
  //! The convexity regularization term.
  double convexity;

  //! The lipschitz constant.
  double lipschitz;

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
};

// Convenience typedefs.

/**
 * Katyusha using the standard update step.
 */
using Katyusha = KatyushaType<false>;

/**
 * Katyusha using the proximal update step.
 */
using KatyushaProximal = KatyushaType<true>;

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "katyusha_impl.hpp"

#endif
