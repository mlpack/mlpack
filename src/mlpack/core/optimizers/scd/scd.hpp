/**
 * @file scd.hpp
 * @author Shikhar Bhardwaj
 *
 * Stochastic Coordinate Descent (SCD).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SCD_SCD_HPP
#define MLPACK_CORE_OPTIMIZERS_SCD_SCD_HPP

#include <mlpack/prereqs.hpp>
#include "descent_policies/random_descent.hpp"

namespace mlpack {
namespace optimization {

/**
 * Stochastic Coordinate descent is a technique for minimizing a function by
 * doing a line search along a single direction at the current point in the
 * iteration. The direction (or "coordinate") can be chosen cyclically, randomly
 * or in a greedy fashion(depending on the DescentPolicy).
 *
 * This optimizer is useful for problems with a smooth multivariate function
 * where computing the entire gradient for an update is infeasable. CD method
 * typically significantly outperform GD, especially on sparse problems with a
 * very large number variables/coordinates.
 *
 * For more information, see the following.
 * @code
 * @inproceedings{Shalev-Shwartz2009,
 *   author    = {Shalev-Shwartz, Shai and Tewari, Ambuj},
 *   title     = {Stochastic Methods for L1 Regularized Loss Minimization},
 *   booktitle = {Proceedings of the 26th Annual International Conference on
 *                Machine Learning},
 *   series    = {ICML '09},
 *   year      = {2009},
 *   isbn = {978-1-60558-516-1}
 * }
 * @endcode
 *
 * For SCD to work, the class must implement the following functions:
 *
 *  size_t NumFeatures();
 *  double Evaluate(const arma::mat& coordinates);
 *  void PartialGradient(const arma::mat& coordinates,
 *                       const size_t j,
 *                       arma::sp_mat& gradient);
 *
 *  NumFeatures() should return the number of features in the decision variable.
 *  Evaluate gives the value of the loss function at the current decision
 *  variable and PartialGradient is used to evaluate the partial gradient with
 *  respect to the jth feature.
 *
 *  @tparam DescentPolicy Descent policy to decide the order in which the
 *      coordinate for descent is selected.
 */
template <typename DescentPolicyType = RandomDescent>
class SCD
{
 public:
  /**
   * Construct the SCD optimizer with the given function and parameters. The
   * default value here are not necessarily good for every problem, so it is
   * suggested that the values used are tailored for the task at hand. The
   * maximum number of iterations refers to the maximum number of "descents"
   * the algorithm does (in one iteration, the algorithm updates the
   * decision variable numFeatures times).
   *
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means to
   *    limit).
   * @param tolerance Maximum absolute tolerance to terminate the algorithm.
   * @param updateInterval The interval at which the objective is to be
   *    reported and checked for convergence.
   * @param descentPolicy The policy to use for picking up the coordinate to
   *    descend on.
   */
  SCD(const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const size_t updateInterval = 1e3,
      const DescentPolicyType descentPolicy = DescentPolicyType());

  /**
   * Optimize the given function using stochastic coordinate descent. The
   * given starting point will be modified to store the finishing point of
   * the optimization, and the final objective value is returned.
   *
   * @tparam ResolvableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value at the final point.
   */
  template <typename ResolvableFunctionType>
  double Optimize(ResolvableFunctionType& function, arma::mat& iterate);

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

  //! Get the update interval for reporting objective.
  size_t UpdateInterval() const { return updateInterval; }
  //! Modify the update interval for reporting objective.
  size_t& UpdateInterval() { return updateInterval; }

  //! Get the descent policy.
  DescentPolicyType DescentPolicy() const { return descentPolicy; }
  //! Modify the descent policy.
  DescentPolicyType& DescentPolicy() { return descentPolicy; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! The update interval for reporting objective and testing for convergence.
  size_t updateInterval;

  //! The descent policy used to pick the coordinates for the update.
  DescentPolicyType descentPolicy;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "scd_impl.hpp"

#endif
