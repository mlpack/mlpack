/**
 * @file inertia_weight.hpp
 * @author Adeel Ahmad
 *
 * PSO with inertia weight.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_INERTIA_WEIGHT_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_INERTIA_WEIGHT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/*
 * PSO with inertia weight.
 */
class InertiaWeight
{
 public:
  /**
   * Implement the PSO with inertia weight. In this variant, the inertia weight
   * is used to calibrate the influence of the previous particle's velocity.
   * The velocity update is given by the following equation:
   * \f[
   * v_{i}(t + 1) = \omega * v_{i}(t) + c1 * rand() * (x_{pbest_{i}}
   * - x_{i}) + c2 * rand() * (x_{gBest_{i}} - x_{i})
   * \f]
   *
   * @param particlePosition Position of the particles.
   * @param particleVelocity Velocity of the particles.
   * @param bestParticlePosition Best position of the particles.
   * @param bestSwarmPosition Best position of the swarm.
   * @param interiaWeight Inertia weight of the particles (omega).
   * @param cognitiveAcceleration Cognitive acceleration of the particles.
   * @param socialAcceleration Social acceleration of the particles.
   */
  double Update(const double particlePosition,
                      const double particleVelocity,
                      const double bestParticlePosition,
                      const double bestSwarmPosition,
                      const double interiaWeight,
                      const double cognitiveAcceleration,
                      const double socialAcceleration)

  {
    return interiaWeight * particleVelocity + cognitiveAcceleration *
        math::Random() * (bestParticlePosition - particlePosition) +
        socialAcceleration * math::Random() *
        (bestSwarmPosition - particlePosition);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
