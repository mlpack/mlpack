/**
 * @file lbest_update.hpp
 *
 * Adam optimizer. Adam is an an algorithm for first-order gradient-based
 * optimization of stochastic objective functions, based on adaptive estimates
 * of lower-order moments.
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
 * Adam is an optimizer that computes individual adaptive learning rates for
 * different parameters from estimates of first and second moments of the
 * gradients as given in the section 7 of the following paper.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Kingma2014,
 *   author  = {Diederik P. Kingma and Jimmy Ba},
 *   title   = {Adam: {A} Method for Stochastic Optimization},
 *   journal = {CoRR},
 *   year    = {2014},
 *   url     = {http://arxiv.org/abs/1412.6980}
 * }
 * @endcode
 */
class LBestUpdate
{
 public:
  /**
   * Construct the velocity update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  LBestUpdate()
  {
    std::cout << "OBJECT CONSTRUCTED" << std::endl;
    // Nothing to do.
  }

  /**
   * The Initialize method is called by PSO Optimizer method before the start of
   * the iteration update process.
   *
   * @param exploitationFactor Influence of personal best achieved.
   * @param explorationFactor Influence of neighbouring particles.
   */
  void Initialize(const double& explorationFactor,
                  const double& exploitationFactor)
  {
    double phi = explorationFactor + exploitationFactor;
    double phiSquared = phi * phi;

    chi = 2 / std::abs(2 - phi - std::sqrt(phiSquared - 4 * phi));
    std::cout << "CHI: " << chi << std::endl;
  }

  /**
   * Update step for Adam.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(const size_t& numParticles,
              const double& exploitationFactor,
              const double& explorationFactor,
              const arma::cube& particlePositions,
              arma::cube& particleVelocities,
              const arma::cube& particleBestPositions,
              const arma::vec& particleBestFitnesses,
              arma::vec& localBestIndices)
  {
    std::cout << "UPDATE CALLED: " << numParticles << std::endl;
    // Content
  }

 private:
  //! Constriction factor chi.
  double chi;
};

} // namespace optimization
} // namespace mlpack

#endif
