/**
 * @file parallel_sgd.hpp
 * @author Shikhar Bhardwaj
 *
 * Parallel Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>
#include "decay_policies/constant_step.hpp"

namespace mlpack {
namespace optimization {

/**
 * An implementation of parallel stochastic gradient descent using the lock-free
 * HOGWILD! approach.
 *
 * For more information, see the following.
 * @misc{1106.5730,
 *   Author = {Feng Niu and Benjamin Recht and Christopher Re and Stephen J.
 *             Wright},
 *   Title = {HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic
 *            Gradient Descent},
 *   Year = {2011},
 *   Eprint = {arXiv:1106.5730},
 * }
 *
 * For Parallel SGD to work, a SparseFunctionType template parameter is
 * required. This class must implement the following functions:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates, const size_t i);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::sp_mat& gradient);
 *
 * In these functions the parameter id refers to which individual function (or
 * gradient) is being evaluated. In case of a data-dependent function, the id
 * would refer to the index of the datapoint(or training example).
 * The data is distributed uniformly among the threads made available to the
 * program by the OpenMP runtime.
 *
 * The Gradient function interface is slightly changed from the
 * DecomposableFunctionType interface, it takes in a sparse matrix as the
 * out-param for the gradient, as ParallelSGD is only expected to be relevant in
 * situations where the computed gradient is sparse.
 *
 * @tparam DecayPolicyType Step size update policy used by parallel SGD
 *     to update the stepsize after each iteration.
 */
template <typename DecayPolicyType = ConstantStep>
class ParallelSGD
{
 public:
  /**
   * Construct the parallel SGD optimizer to optimize the given function with
   * the given parameters. One iteration means one batch of datapoints processed
   * by each thread.
   *
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.
   *
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param threadShareSize Number of datapoints to be processed in one
   *     iteration by each thread.
   * @param tolerance Maximum absolute tolerance to terminate the algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   * @param decayPolicy The step size update policy to use.
  */
  ParallelSGD(const size_t maxIterations,
              const size_t threadShareSize,
              const double tolerance = 1e-5,
              const bool shuffle = true,
              const DecayPolicyType& decayPolicy = DecayPolicyType());

  /**
   * Optimize the given function using the parallel SGD algorithm. The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the value of the loss function at the final point is
   * returned.
   *
   * @tparam SparseFunctionType Type of function to be optimized.
   * @param function Function to be optimized(minimized).
   * @param iterate Starting point(will be modified).
   * @return Objective value at the final point.
   */
  template <typename SparseFunctionType>
  double Optimize(SparseFunctionType& function, arma::mat& iterate);

  //! Get the maximum number of iterations (0 indicates no limits).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limits).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the number of datapoints to be processed in one iteration by each
  //! thread.
  size_t ThreadShareSize() const { return threadShareSize; }
  //! Modify the number of datapoints to be processed in one iteration by each
  //! thread.
  size_t& ThreadShareSize() { return threadShareSize; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

  //! Get the step size decay policy.
  DecayPolicyType& DecayPolicy() const { return decayPolicy; }
  //! Modify the step size decay policy.
  DecayPolicyType& DecayPolicy() { return decayPolicy; }

 private:
  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The number of datapoints to be processed in one iteration by each thread.
  size_t threadShareSize;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;

  //! The step size decay policy.
  DecayPolicyType decayPolicy;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
