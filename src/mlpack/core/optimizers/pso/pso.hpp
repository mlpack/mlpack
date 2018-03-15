/**
 * @file pso.hpp
 * @author Chintan Soni
 *
 * Particle swarm optimization using the lbest approach.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_PSO_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_PSO_HPP

#include <mlpack/core.hpp>

#include "lbest_update.hpp"
#include <iostream>

namespace mlpack {
namespace optimization {

/**
 * Gradient Descent is a technique to minimize a function. To find a local
 * minimum of a function using gradient descent, one takes steps proportional
 * to the negative of the gradient of the function at the current point,
 * producing the following update scheme:
 *
 * \f[
 * A_{j + 1} = A_j + \alpha \nabla F(A)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size. \f$ F \f$
 * is the function being optimized. The algorithm continues until \f$ j
 * \f$ reaches the maximum number of iterations---or when an update produces
 * an improvement within a certain tolerance \f$ \epsilon \f$.  That is,
 *
 * \f[
 * | F(A_{j + 1}) - F(A_j) | < \epsilon.
 * \f]
 *
 * The parameter \f$\epsilon\f$ is specified by the tolerance parameter to the
 * constructor.
 *
 * For Gradient Descent to work, a FunctionType template parameter is required.
 * This class must implement the following function:
 *
 *   double Evaluate(const arma::mat& coordinates);
 *   void Gradient(const arma::mat& coordinates,
 *                 arma::mat& gradient);
 */
template<typename VelocityUpdatePolicy = LBestUpdate>
class PSOType
{
 public:
  /**
   * Construct the Gradient Descent optimizer with the given function and
   * parameters.  The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task
   * at hand.
   *
   * @param function Function to be optimized (minimized).
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   */
  PSOType(const size_t numParticles = 16,
          const size_t maxIterations = 100000,
          const double exploitationFactor = 2.05,
          const double explorationFactor = 2.05,
          const VelocityUpdatePolicy& velocityUpdatePolicy = VelocityUpdatePolicy()) :
          numParticles(numParticles),
          maxIterations(maxIterations),
          exploitationFactor(exploitationFactor),
          explorationFactor(explorationFactor),
          velocityUpdatePolicy(velocityUpdatePolicy) { /*Nothing to do */ }

  /**
   * Optimize the given function using gradient descent.  The given starting
   * point will be modified to store the finishing point of the algorithm, and
   * the final objective value is returned.
   *
   * @tparam FunctionType Type of the function to optimize.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename FunctionType>
  double Optimize(FunctionType& function, arma::mat& iterate);

  //! Retrieve value of numParticles.
  size_t NumParticles() const { return numParticles; }

  //! Modify value of numParticles.
  size_t& NumParticles() { return numParticles; }

private:
  //! Number of particles in the swarm.
  size_t numParticles;
  //! Maximum number of iterations for which the optimizer will run.
  size_t maxIterations;
  //! Exploitation factor for lbest version.
  double exploitationFactor;
  //! Exploration factor for lbest version.
  double explorationFactor;
  //! Particle positions.
  arma::cube particlePositions;
  //! Particle velocities.
  arma::cube particleVelocities;
  //! Best fitness attained by particle so far.
  arma::vec particleBestFitnesses;
  //! Position corresponding to the best fitness of particle.
  arma::cube particleBestPositions;
  //! Index of the best neighbour.
  arma::vec localBestIndices;
  //! Velocity update policy used.
  VelocityUpdatePolicy velocityUpdatePolicy;
};

using LBestPSO = PSOType<LBestUpdate>;

} // namespace optimization
} // namespace mlpack

#include "pso_impl.hpp"

#endif
