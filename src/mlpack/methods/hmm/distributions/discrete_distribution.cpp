/**
 * @file discrete_distribution_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the discrete distribution.
 */
#ifndef __MLPACK_METHODS_HMM_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_IMPL_HPP
#define __MLPACK_METHODS_HMM_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_IMPL_HPP

// Just in case.
#include "discrete_distribution.hpp"

namespace mlpack {
namespace distribution {

// These functions are inlined because they are so simple.
DiscreteDistribution::DiscreteDistribution(size_t numObservations)
    : probability(numObservations)
{ /* nothing to do */ }

inline double DiscreteDistribution::Probability(size_t observation)
{
  // No bounds checking for speed reasons.
  return probability(observation);
}

}; // namespace distribution
}; // namespace mlpack

#endif
