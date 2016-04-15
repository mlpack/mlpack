/**
 * @file gaussian_distribution.hpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Implementation of the Gaussian distribution.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace distribution {

/**
 * A single multivariate Gaussian distribution.
 */
class MLPACK_API GaussianDistribution
{
 private:
  //! Mean of the distribution.
  arma::vec mean;
  //! Positive definite covariance of the distribution.
  arma::mat covariance;
  //! Lower triangular factor of cov (e.g. cov = LL^T).
  arma::mat covLower;
  //! Cached inverse of covariance.
  arma::mat invCov;
  //! Cached logdet(cov).
  double logDetCov;

  //! log(2pi)
  static const constexpr double log2pi = 1.83787706640934533908193770912475883;

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
      covariance(arma::eye<arma::mat>(dimension, dimension)),
      covLower(arma::eye<arma::mat>(dimension, dimension)),
      invCov(arma::eye<arma::mat>(dimension, dimension)),
      logDetCov(0)
  { /* Nothing to do. */ }

  /**
   * Create a Gaussian distribution with the given mean and covariance.
   *
   * covariance is expected to be positive definite.
   */
  GaussianDistribution(const arma::vec& mean, const arma::mat& covariance);

  // TODO(stephentu): do we want a (arma::vec&&, arma::mat&&) ctor?

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  /**
   * Return the probability of the given observation.
   */
  double Probability(const arma::vec& observation) const
  {
    return exp(LogProbability(observation));
  }

  /**
   * Return the log probability of the given observation.
   */
  double LogProbability(const arma::vec& observation) const;

  /**
   * Calculates the multivariate Gaussian probability density function for each
   * data point (column) in the given matrix
   *
   * @param x List of observations.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const arma::mat& x, arma::vec& probabilities) const
  {
    arma::vec logProbabilities;
    LogProbability(x, logProbabilities);
    probabilities = arma::exp(logProbabilities);
  }

  /**
  * Calculates the multivariate Gaussian log probability density function for each
  * data point (column) in the given matrix
  *
  * @param x List of observations.
  * @param probabilities Output log probabilities for each input observation.
  */
  void LogProbability(const arma::mat& x, arma::vec& logProbabilities) const;

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
  void Train(const arma::mat& observations);

  /**
   * Estimate the Gaussian distribution from the given observations, taking into
   * account the probability of each observation actually being from this
   * distribution.
   */
  void Train(const arma::mat& observations,
             const arma::vec& probabilities);

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
   * Set the covariance.
   */
  void Covariance(const arma::mat& covariance);

  void Covariance(arma::mat&& covariance);

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using data::CreateNVP;

    // We just need to serialize each of the members.
    ar & CreateNVP(mean, "mean");
    ar & CreateNVP(covariance, "covariance");
    ar & CreateNVP(covLower, "covLower");
    ar & CreateNVP(invCov, "invCov");
    ar & CreateNVP(logDetCov, "logDetCov");
  }

 private:
  /**
   * This factors the covariance using arma::chol().  The function assumes that
   * the given matrix is factorizable via the Cholesky decomposition.  If not, a
   * std::runtime_error will be thrown.
   */
  void FactorCovariance();
};


} // namespace distribution
} // namespace mlpack

#endif
