/**
 * @file gaussian_distribution.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Gaussian distribution.
 */
#ifndef __MLPACK_METHODS_HMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP
#define __MLPACK_METHODS_HMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP

#include <mlpack/core.hpp>
// Should be somewhere else, maybe in core.
#include <mlpack/methods/gmm/phi.hpp>

namespace mlpack {
namespace distribution {

class GaussianDistribution
{
 private:
  //! Mean of the distribution.
  arma::vec mean;
  //! Covariance of the distribution.
  arma::mat covariance;

 public:
  //! The type of data which this distribution uses.
  typedef arma::vec DataType;

  /**
   * Default constructor, which creates a Gaussian with zero dimension.
   */
  GaussianDistribution() { /* nothing to do */ }

  /**
   * Create a Gaussian distribution with zero mean and identity covariance with
   * the given dimensionality.
   */
  GaussianDistribution(const size_t dimension) :
      mean(dimension), covariance(arma::eye<arma::mat>(dimension, dimension))
  { /* nothing to do */ }

  /**
   * Create a Gaussian distribution with the given mean and covariance.
   */
  GaussianDistribution(const arma::vec& mean, const arma::mat& covariance) :
      mean(mean), covariance(covariance) { /* nothing to do */ }

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
  void Estimate(const std::vector<arma::vec> observations);

  /**
   * Estimate the Gaussian distribution from the given observations, taking into
   * account the probability of each observation actually being from this
   * distribution.
   */
  void Estimate(const std::vector<arma::vec> observations,
                const std::vector<double> probabilities);

  /**
   * Return the mean.
   */
  const arma::vec& Mean() const { return mean; }

  /**
   * Return a modifiable copy of the mean.
   */
  arma::vec& Mean() { return mean; }

  /**
   * Return the covariance matrix.
   */
  const arma::mat& Covariance() const { return covariance; }

  /**
   * Return a modifiable copy of the covariance.
   */
  arma::mat& Covariance() { return covariance; }

};

}; // namespace distribution
}; // namespace mlpack

#endif
