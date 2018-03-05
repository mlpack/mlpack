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
    double phi = cognitiveAcceleration + socialAcceleration;
    k = 2 / std::abs(2 - phi - std::sqrt(std::pow(phi, 2) - 4 * phi));

    for (size_t i = 0; i < dimension; ++i)
    {
      for (size_t j = 0; j < particlePosition.slice(0).n_rows; ++j)
      {
        for (size_t k = 0; k < particlePosition.slice(0).n_cols; ++k)
        {
          particleVelocity.slice(i)[j, k] = k * (interiaWeight *
            particleVelocity.slice(i)[j, k] + cognitiveAcceleration *
            math::Random() * (bestParticlePosition[i] - particlePosition
            .slice(i)[j, k]) + socialAcceleration * math::Random() *
            (bestSwarmPosition[i] - particlePosition.slice(i)[j, k]));
        }
      }
    }
  }

  private:
    // The constriction factor.
    double k;
};

} // namespace optimization
} // namespace mlpack

#endif
