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
 * out-param for the gradient. As ParallelSGD is only expected to be relevant in
 * situations where the computed gradient is sparse.
 *
 * @tparam DecayPolicyType Step size update policy used by parallel SGD
 *     to update the stepsize after each iteration.
 */
template <typename DecayPolicyType>
class ParallelSGD
{
 public:
  /**
   * Construct the parallel SGD optimizer to optimize the given function with
   * the given parameters. One iteration means one batch of datapoints processed
   * by each thread.
   *
   * @param maxIterations Maximum number of iterations allowed.
   * @param batchSize Number of datapoints to be processed in one iteration by
   *     each thread.
   * @param tolerance Maximum absolute tolerance to terminate the algorithm.
   * @param decayPolicy The step size update policy to use.
  */
  ParallelSGD(const size_t maxIterations,
              const size_t batchSize,
              const double tolerance,
              const DecayPolicyType& decayPolicy);

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
  size_t BatchSize() const { return batchSize; }
  //! Modify the number of datapoints to be processed in one iteration by each
  //! thread.
  size_t& BatchSize() { return batchSize; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get the step size decay policy.
  DecayPolicyType& DecayPolicy() const { return decayPolicy; }
  //! Modify the step size decay policy.
  DecayPolicyType& DecayPolicy() { return decayPolicy; }

 private:
  /**
   * Generate the indices to be visited by each thread before iteration.
   * Generates a randomly shuffled vector of datapoint indices (range 0 to
   * function.NumFunctions() - 1).
   *
   * @param visitationOrder Out param with the indices of the datapoints for the
   *    current iteration.
   * @param numFunctions The number of separable functions in the objective.
   */
  void GenerateVisitationOrder(arma::Col<size_t>& visitationOrder,
      size_t numFunctions);

  /**
   * Get the share of datapoint indices to be updated by the thread with given
   * thread id.
   *
   * @param thread_id The id of the current thread. Range 0-OMP_NUM_THREADS.
   * @param visitationOrder The random list of datapoint indices for the current
   *    iteration.
   * @return Vector of datapoint indices to be visited by the current thread.
   */
  arma::Col<size_t> ThreadShare(size_t threadId,
                                const arma::Col<size_t>& visitationOrder);

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The number of datapoints to be processed in one iteration by each thread.
  size_t batchSize;

  //! The tolerance for termination.
  double tolerance;

  //! The step size decay policy.
  DecayPolicyType decayPolicy;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
