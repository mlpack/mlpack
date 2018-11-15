/**
 * @file exponential_backoff.hpp
 * @author Shikhar Bhardwaj
 *
 * Exponential backoff step size decay policy for parallel Stochastic Gradient
 * Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PARALLEL_SGD_EXP_BACKOFF_HPP
#define ENSMALLEN_PARALLEL_SGD_EXP_BACKOFF_HPP

namespace ens {

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
class ExponentialBackoff
{
 public:
  /**
   * Member initializer constructor to construct the exponential backoff policy
   * with the required parameters.
   *
   * @param firstBackoffEpoch The number of updates to run before the first
   * stepsize backoff.
   * @param step The initial stepsize(gamma).
   * @param beta The reduction factor. This should be a value in range (0, 1).
   */
  ExponentialBackoff(const size_t firstBackoffEpoch,
                     const double step,
                     const double beta) :
    firstBackoffEpoch(firstBackoffEpoch),
    cutoffEpoch(firstBackoffEpoch),
    step(step),
    beta(beta)
  { /* Nothing to do. */ }

  /**
   * Get the step size for the current gradient update.
   *
   * @param numEpoch The iteration number of the current update.
   * @return The stepsize for the current iteration.
   */
  double StepSize(const size_t numEpoch)
  {
    if (numEpoch >= cutoffEpoch)
    {
      step *= beta;
      cutoffEpoch += firstBackoffEpoch / beta;
    }
    return step;
  }

 private:
  //! The first iteration at which the stepsize should be reduced.
  size_t firstBackoffEpoch;

  //! The iteration at which the next decay will be performed.
  size_t cutoffEpoch;

  //! The initial stepsize.
  double step;

  //! The reduction factor, should be in range (0, 1).
  double beta;
};

} // namespace ens

#endif
