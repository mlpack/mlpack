/**
 * @file gaussian_distribution.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Gaussian distribution.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
