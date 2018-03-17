/**
 * @file lbest_update.hpp
 *
 * ADD LBEST DESCRIPTION HERE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_LBEST_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_LBEST_UPDATE_HPP

#include <mlpack/prereqs.hpp>

#include <iostream>

namespace mlpack {
namespace optimization {

/**
 * ADD SLIGHTLY MORE DETAILED DESCRIPTION HERE.
 *
 * For more information, see the following.
 *
 * REFERENCE TO A RELEVANT ARTICLE.
 *
 */
class LBestUpdate
{
 public:
  /**
   * The Initialize method is called by PSO Optimizer method before the start of
   * the iteration process. It calculates the value of the constriction
   * coefficent, initializes the local best indices of each particle to itself,
   * and sets the shape of the r1 and r2 vectors.
   *
   * @param exploitationFactor Influence of personal best achieved.
   * @param explorationFactor Influence of neighbouring particles.
   * @param numParticles The number of particles in the swarm.
   * @param iterate The user input, used for shaping intermediate vectors.
   */
  void Initialize(const double& exploitationFactor,
                  const double& explorationFactor,
                  const size_t& numParticles,
                  const arma::mat& iterate)
  {
    // Set number of particles.
    n = numParticles;
    // Set c1 = exploitationFactor and c2 = explorationFactor.
    c1 = exploitationFactor;
    c2 = explorationFactor;

    // Calculate the constriction factor
    double phi = c1 + c2;
    assert(phi > 4.0);
    chi = 2.0 / std::abs(2.0 - phi - std::sqrt((phi - 4.0) * phi));

    // Initialize local best indices to self indices of particles.
    size_t index = 0;
    localBestIndices.set_size(n);
    localBestIndices.imbue([&]() { return index++; });

    // Set sizes r1 and r2.
    r1.set_size(iterate.n_rows, iterate.n_cols);
    r2.set_size(iterate.n_rows, iterate.n_cols);
  }

  /**
   * Update step for LBestPSO.
   *
   * ADD DESCRIPTIONS OF RELEVANT PARAMETERS HERE.
   *
   * @param particlePositions The current coordinates of particles.
   * @param particleVelocities The current velocities (will be modified).
   * @param particleFitnesses The current fitness values or particles.
   * @param particleBestPositions The personal best coordinates of particles.
   * @param particleBestFitnesses The personal best fitness values of particles.
   */
  void Update(const arma::cube& particlePositions,
              arma::cube& particleVelocities,
              const arma::vec& particleFitnesses,
              const arma::cube& particleBestPositions,
              const arma::vec& particleBestFitnesses)
  {
    // Velocity update logic.
    for (size_t i = 0; i < n; i++)
    {
      localBestIndices(i) =
        particleBestFitnesses(left(i)) < particleBestFitnesses(i) ?
        left(i) : i;
      localBestIndices(i) =
        particleBestFitnesses(right(i)) < particleBestFitnesses(i) ?
        right(i) : i;
    }

    for (size_t i = 0; i < n; i++)
    {
      // Generate random numbers for current particle.
      r1.randu();
      r2.randu();
      particleVelocities.slice(i) = chi * (particleVelocities.slice(i) +
        c1 * r1 %
          (particleBestPositions.slice(i) - particlePositions.slice(i)) +
        c2 * r2 %
          (particleBestPositions.slice(localBestIndices(i)) -
            particlePositions.slice(i)));
    }
  }

 private:
  //! Number of particles.
  size_t n;
  //! Exploitation factor.
  double c1;
  //! Exploration factor.
  double c2;
  //! Constriction factor chi.
  double chi;
  //! Vectors of random numbers.
  arma::mat r1;
  arma::mat r2;
  //! Indices of each particle's best neighbour.
  arma::vec localBestIndices;

  // Helper functions for calculating neighbours.
  inline const size_t left(const size_t index) { return (index + n - 1) % n; }
  inline const size_t right(const size_t index) { return (index + 1) % n; }
};

} // namespace optimization
} // namespace mlpack

#endif
