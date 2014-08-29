/**
 * @file gaussian_distribution.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Gaussian distribution.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_HMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP
#define __MLPACK_METHODS_HMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP

#include <mlpack/core.hpp>
// Should be somewhere else, maybe in core.
#include <mlpack/methods/gmm/phi.hpp>

namespace mlpack {
namespace distribution {

/**
 * A single multivariate Gaussian distribution.
 */
class GaussianDistribution
{
 private:
  //! Mean of the distribution.
  arma::vec mean;
  //! Covariance of the distribution.
  arma::mat covariance;

 public:
  /**
   * Default constructor, which creates a Gaussian with zero dimension.
   */
  GaussianDistribution() { /* nothing to do */ }

  /**
   * Create a Gaussian distribution with zero mean and identity covariance with
   * the given dimensionality.
   */
  GaussianDistribution(const size_t dimension) :
      mean(arma::zeros<arma::vec>(dimension)),
      covariance(arma::eye<arma::mat>(dimension, dimension))
  { /* Nothing to do. */ }

  /**
   * Create a Gaussian distribution with the given mean and covariance.
   */
  GaussianDistribution(const arma::vec& mean, const arma::mat& covariance) :
      mean(mean), covariance(covariance) { /* Nothing to do. */ }

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  /**
   * Return the probability of the given observation.
   */
  double Probability(const arma::vec& observation) const
  {
    return mlpack::gmm::phi(observation, mean, covariance);
  }

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this Gaussian distribution.
   */
  arma::vec Random() const;

  /**
   * Estimate the Gaussian distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Estimate(const arma::mat& observations);

  /**
   * Estimate the Gaussian distribution from the given observations, taking into
   * account the probability of each observation actually being from this
   * distribution.
   */
  void Estimate(const arma::mat& observations,
                const arma::vec& probabilities);

  //! Return the mean.
  const arma::vec& Mean() const { return mean; }
  //! Return a modifiable copy of the mean.
  arma::vec& Mean() { return mean; }

  //! Return the covariance matrix.
  const arma::mat& Covariance() const { return covariance; }
  //! Return a modifiable copy of the covariance.
  arma::mat& Covariance() { return covariance; }

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;
};

}; // namespace distribution
}; // namespace mlpack

#endif
