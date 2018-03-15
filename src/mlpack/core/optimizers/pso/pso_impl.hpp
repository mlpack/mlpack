/**
 * @file pso_impl.hpp
 * @author Chintan Soni
 *
 * Implementation of the lbest particle swarm optimization algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_PSO_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_PSO_IMPL_HPP

#include "pso.hpp"

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

//! Optimize the function (minimize).
template<typename VelocityUpdatePolicy>
template<typename FunctionType>
double PSOType<VelocityUpdatePolicy>::Optimize(FunctionType& function, arma::mat& iterate)
{
  // The following cast and checks are added for the sake of extending the PSO
  // algorithm with Gradient Descent and creating a hybrid optimizer.

  // To ensure that we have all the necessary functions.
  typedef Function<FunctionType> FullFunctionType;
  FullFunctionType& f(static_cast<FullFunctionType&>(function));

  // Make sure we have the methods that we need.
  traits::CheckFunctionTypeAPI<FullFunctionType>();

  // Set a random seed for the random number generator.
  arma::arma_rng::set_seed_random();

  // Randomly initialize the particle positions and velocities.
  particlePositions.randu(iterate.n_rows, iterate.n_cols, numParticles);
  particleVelocities.randu(iterate.n_rows, iterate.n_cols, numParticles);

  // Copy to personal best values for first iteration.
  particleBestPositions = particlePositions;
  // Initialize personal best fitness values to infinity.
  particleBestFitnesses.fill(std::numeric_limits<double>::max());

  // Initialize local best indices to self indices of particles.
  size_t index = 0;
  localBestIndices.set_size(numParticles);
  localBestIndices.imbue([&]() { return index++; });

  // Initialize the update policy.
  velocityUpdatePolicy.Initialize(2.05, 2.05);

  velocityUpdatePolicy.Update(numParticles,
                              exploitationFactor,
                              explorationFactor,
                              particlePositions,
                              particleVelocities,
                              particleBestPositions,
                              particleBestFitnesses,
                              localBestIndices);

  std::cout << "OPTIMIZER CALLED" << std::endl;
  std::cout << "CHANGES MADE" << std::endl;
  return 0.0;
  // return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
