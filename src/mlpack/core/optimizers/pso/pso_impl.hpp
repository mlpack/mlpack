/**
 * @file pso_impl.hpp
 * @author Adeel Ahmad
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
                                  const size_t dimension,
                                  const double interiaWeight,
                                  const double cognitiveAcceleration,
                                  const double socialAcceleration,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const VelocityVectorType& velocityType) :
    lambda(lambda),
    dimension(dimension),
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
  // Initialize the particle position and velocity. Following a
  // heuristic the swarm size is set to 2 * dimension.
  lambda = 2 * dimension;
  particlePosition.randu(lambda, dimension);
  particleVelocity.randu(lambda, dimension);
  bestParticlePosition.randu(lambda, dimension);
  bestSwarmPosition.randu(lambda, dimension);

  // Calculate the first objective function.
  double currentObjective = function.Evaluate(iterate);

  double overallObjective = currentObjective;
  double lastObjective = DBL_MAX;

  // Update the position and velocity for each particle.
  for (size_t i = 0; i < maxIterations; ++i)
  {
    for (size_t j = 0; j < lambda; ++j)
    {
      for (size_t k = 0; k < dimension; ++k)
      {
        // Update velocity.
        particleVelocity(j, k) = velocityType.Update(particlePosition(j, k),
          particleVelocity(j, k), bestParticlePosition(j, k),
          bestSwarmPosition(j, k), interiaWeight,
          cognitiveAcceleration, socialAcceleration);

        // Update position.
        particlePosition(j, k) += particleVelocity(j, k);

        // Evaluate the objective function.
        currentObjective = function.Evaluate(particleVelocity);

        // Update best parameters.
        if (currentObjective < overallObjective)
        {
          overallObjective = currentObjective;
        }

        // Compare objective with tolerance.
        if (std::abs(lastObjective - overallObjective) < tolerance)
        {
          Log::Info << "PSO: minimized within tolerance " << tolerance << "; "
              << "terminating optimization." << std::endl;
          return overallObjective;
        }
      }
    }
    lastObjective = currentObjective;
  }
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
