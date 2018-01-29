/**
 * @file pso_impl.hpp
 *
 * Implementation of the Particle Swarm Optimizer as proposed
 * by J. Kennedy et al. in "Particle swarm optimization".
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_PSO_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_PSO_IMPL_HPP

// In case it hasn't been included yet.
#include "pso.hpp"

namespace mlpack {
namespace optimization {

template<typename VelocityVectorType>
PSO<VelocityVectorType>::PSO(const size_t lambda,
                                  const arma::mat particlePosition,
                                  const arma::mat particleVelocity,
                                  const arma::mat bestParticlePosition,
                                  const double bestSwarmPosition,
                                  const double interiaWeight,
                                  const double cognitiveAcceleration,
                                  const double socialAcceleration,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const VelocityVectorType& velocityType) :
    lambda(lambda),
    particlePosition(particlePosition),
    particleVelocity(particleVelocity),
    bestParticlePosition(bestParticlePosition),
    bestSwarmPosition(bestSwarmPosition),
    interiaWeight(interiaWeight),
    cognitiveAcceleration(cognitiveAcceleration),
    socialAcceleration(socialAcceleration),
    tolerance(tolerance),
    velocityType(velocityType)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename VelocityVectorType>
template<typename DecomposableFunctionType>
double PSO<VelocityVectorType>::Optimize(
    DecomposableFunctionType& function, arma::mat& iterate)
{
  // Start iterating.
  for (size_t i = 1; i < maxIterations; ++i)
  {
    for (size_t j = 0; j < lambda; ++j)
    {
       // TODO: Implement the algorithm
    }
  }

}

} // namespace optimization
} // namespace mlpack

#endif
