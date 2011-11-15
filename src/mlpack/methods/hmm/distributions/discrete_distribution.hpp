/**
 * @file discrete_distribution.hpp
 * @author Ryan Curtin
 *
 * Implementation of the discrete distribution, where each discrete observation
 * has a given probability.
 */
#ifndef __MLPACK_METHODS_HMM_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP
#define __MLPACK_METHODS_HMM_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP

#include <mlpack/core.h>

namespace mlpack {
namespace distribution {

/**
 * A discrete distribution where the only observations are of type size_t.  This
 * is useful (for example) with discrete Hidden Markov Models, where
 * observations are non-negative integers representing specific emissions.
 *
 * No bounds checking is performed for observations, so if an invalid
 * observation is passed (i.e. observation > numObservations), a crash will
 * probably occur.
 */
class DiscreteDistribution
{
 public:
  /**
   * Define the discrete distribution as having numObservations possible
   * observations.  The probability in each state will be set to (1 /
   * numObservations).
   *
   * @param numObservations Number of possible observations this distribution
   *    can have.
   */
  DiscreteDistribution(size_t numObservations);

  /**
   * Return the probability of the given observation.  If the observation is
   * greater than the number of possible observations, then a crash will
   * probably occur -- bounds checking is not performed.
   *
   * @param observation Observation to return the probability of.
   * @return Probability of the given observation.
   */
  double Probability(size_t observation);

 private:
  arma::vec probability;
};

// Include inline implementation.
#include "discrete_distribution_impl.hpp"

}; // namespace distribution
}; // namespace mlpack

#endif
