/**
 * @file exponential_backoff.hpp
 * @author Shikhar Bhardwaj
 *
 * Exponential backoff step size decay policy for parallel Stochastic Gradient
 * Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_EXP_BACKOFF_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_EXP_BACKOFF_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace optimization{

/**
 * Exponential backoff stepsize reduction policy for parallel SGD.
 *
 * For more information, see the following.
 *
 * @misc{1106.5730,
 *   Author = {Feng Niu and Benjamin Recht and Christopher Re and Stephen J.
 *             Wright},
 *   Title = {HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic
 *            Gradient Descent},
 *   Year = {2011},
 *   Eprint = {arXiv:1106.5730},
 * }
 *
 * This stepsize update scheme gives robust 1/k convergence rates to the
 * implementation of parallel SGD.
 */
class ExponentialBackoff{
 public:
  /**
   * Construct the exponential backoff policy with the required parameters.
   *
   * @param firstBackoffEpoch The number of updates to run before the first
   * stepsize backoff.
   * @param step The initial stepsize(gamma).
   * @param beta The reduction factor.
   */
  ExponentialBackoff(size_t firstBackoffEpoch, double step, double beta) :
    firstBackoffEpoch(firstBackoffEpoch), step(step), beta(beta)
  {
      cutoffEpoch = firstBackoffEpoch;
  }
  /**
   * Get the step size for the current gradient update.
   *
   * @param n_epoch The iteration number of the current update.
   */
  double StepSize(size_t n_epoch)
  {
    if (n_epoch >= cutoffEpoch)
    {
      step /= beta;
      cutoffEpoch += firstBackoffEpoch / beta;
    }
    return step;
  }

 private:
  size_t firstBackoffEpoch, cutoffEpoch;
  double step, beta;
};

} // namespace optimization
} // namespace mlpack

#endif
