/**
 * @file default_init.hpp
 *
 * The default initialization policy used by the PSO optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_INIT_POLICIES_DEFAULT_INIT_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_INIT_POLICIES_DEFAULT_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <iostream>

namespace mlpack {
namespace optimization {

/**
 * The default initialization policy used by the PSO optimizer. It initializes
 * particle positions uniformly in [-1, 1], the velocities in [0, 1] personal
 * bests of the particles to the initial positions, and all fitness values to
 * std::numeric_limits<double>::max().
 */
class DefaultInit
{
 public:
  /**
   * Constructor for the DefaultInit policy. The policy initializes particle
   * posiitons in the range [lowerBound, upperBound]. Defaults to [-1, 1].
   *
   * @param lowerBound Lower bound of the position initialization range.
   * @param upperBound Upper bound of the position initialization range.
   */
  DefaultInit(const double lowerBound = -1.0,
              const double upperBound = 1.0):
      lowerBound(lowerBound),
      upperBound(upperBound)
  {
    /* Nothing to do */
  }

  /**
   * The InitializeParticles method of the init policy. Any class that is used
   * in place of this default must implement this method which is used by the
   * optimizer.
   *
   * @param iterate Coordinates of the initial point for training.
   * @param numParticles The number of particles in the swarm.
   * @param particlePositions Current positions of particles.
   * @param particleVelocities Current velocities of particles.
   * @param particleFitnesses Current fitness values of particles.
   * @param particleBestPositions Best positions attained by each particle.
   * @param particleBestFitnesses Best fitness values attained by each particle.
   */
  void InitializeParticles(const arma::mat& iterate,
                           const size_t numParticles,
                           arma::cube& particlePositions,
                           arma::cube& particleVelocities,
                           arma::mat& particleFitnesses,
                           arma::cube& particleBestPositions,
                           arma::mat& particleBestFitnesses)
  {
    // Randomly initialize the particle positions.
    particlePositions.randu(iterate.n_rows, iterate.n_cols, numParticles);

    // Distribute particles in [lowerBound, upperBound].
    particlePositions *= upperBound - lowerBound;
    particlePositions += lowerBound;

    // Randomly initialize particle velocities.
    particleVelocities.randu(iterate.n_rows, iterate.n_cols, numParticles);

    // Initialize current fitness values to infinity.
    particleFitnesses.set_size(numParticles);
    particleFitnesses.fill(std::numeric_limits<double>::max());

    // Copy to personal best values for first iteration.
    particleBestPositions = particlePositions;
    // Initialize personal best fitness values to infinity.
    particleBestFitnesses.set_size(numParticles);
    particleBestFitnesses.fill(std::numeric_limits<double>::max());
  }

  //! Retrieve value of lowerBound.
  double LowerBound() const { return lowerBound; }

  //! Modify value of lowerBound.
  double& LowerBound() { return lowerBound; }

  //! Retrieve value of upperBound.
  double UpperBound() const { return upperBound; }

  //! Modify value of upperBound.
  double& UpperBound() { return upperBound; }

 private:
  //! Lower bound.
  double lowerBound;
  //! Upper bound.
  double upperBound;
};

} // namespace optimization
} // namespace mlpack

#endif
