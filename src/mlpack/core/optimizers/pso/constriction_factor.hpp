/**
 * @file constriction_factor.hpp
 * @author Adeel Ahmad
 *
 * PSO with constriction factor.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_CONSTRICTION_FACTOR_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_CONSTRICTION_FACTOR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/*
 * PSO with constriction factor.
 */
class ConstrictionFactor
{
 public:
  /**
   * Implement the PSO with constriction factor. In this variant, the
   * constriction factor \f$ k \f$ is included to the veloctiy formula.
   * It is calculated using the acceleration constants \f$ c1 \f$ and
   * \f$ c2 \f$. The updated veloctiy equation is given by:
   *
   * \f[
   * v_{i}(t + 1) = k * [\omega * v_{i}(t) + c1 * rand() * (x_{pbest_{i}}
   * - x_{i}) + c2 * rand() * (x_{gBest_{i}} - x_{i})]
   * \f]
   * 
   * The constriction factor \f$ k \f$ is given by:
   *
   * \f[
   * k = \frac{2}{\left | 2 - \phi - \sqrt{\phi^{2} - 4\phi } \right |}
   * \f]
   *
   * where \f$ \phi = c1 + c2, \phi > 4 \f$.
   * 
   * The constriction factor approach can generate higher
   * quality solutions than the basic PSO approach.
   *
   * For more information, please refer to:
   *
   * @code
   * @article{Lim2009
   *   author    = {Lim, S., Montakhab, M. and Nouri, H.},
   *   title     = {A constriction factor based particle swarm optimization for economic dispatch},
   *   year      = {2009},
   *   pages     = {4-7},
   * }
   * @endcode
   * 
   * @param particlePosition Position of the particles.
   * @param particleVelocity Velocity of the particles.
   * @param bestParticlePosition Best position of the particles.
   * @param bestSwarmPosition Best position of the swarm.
   * @param interiaWeight Inertia weight of the particles (omega).
   * @param cognitiveAcceleration Cognitive acceleration of the particles.
   * @param socialAcceleration Social acceleration of the particles.
   */
  void Update(const arma::cube& particlePosition,
              arma::cube& particleVelocity,
              const arma::mat& bestParticlePosition,
              const arma::mat& bestSwarmPosition,
              const double& interiaWeight,
              const double& cognitiveAcceleration,
              const double& socialAcceleration,
              const double& dimension)
  {
    // Calculate the constriction factor.
    double phi = cognitiveAcceleration + socialAcceleration;
    constrictionFactor = 2 / std::abs(2 - phi - std::sqrt(
      std::pow(phi, 2) - 4 * phi));

    // Generate uniform random numbers for velocity updation.
    arma::mat r1(particlePosition.n_rows, particlePosition.n_cols,
      arma::fill::randu);
    arma::mat r2(particlePosition.n_rows, particlePosition.n_cols,
      arma::fill::randu);

    for (size_t i = 0; i < dimension; ++i)
    {
      particleVelocity.slice(i) = constrictionFactor * (interiaWeight *
        particleVelocity.slice(i) + cognitiveAcceleration * r1 %
        (bestParticlePosition[i] - particlePosition.slice(i)) +
        socialAcceleration * r2 % (bestSwarmPosition[i] -
        particlePosition.slice(i)));
    }
  }

 private:
  //! The constriction factor.
  double constrictionFactor;
};

} // namespace optimization
} // namespace mlpack

#endif
