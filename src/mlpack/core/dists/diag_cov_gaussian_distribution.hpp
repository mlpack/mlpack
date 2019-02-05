/**
 * @file diag_gaussian_distribution.hpp
 * @author Kim SangYeon
 *
 * Implementation of the Gaussian distribution with diagonal covariance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_DIAG_COV_GAUSSIAN_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_DIAG_COV_GAUSSIAN_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace distribution {

/**
 * A single multivariate Gaussian distribution with diagonal covariance.
 */
class DiagCovGaussianDistribution
{
 private:
  //! Mean of the distribution.
  arma::vec mean;
  //! Diagonal covariance of the distribution.
  arma::mat covariance;
  //! Cached inverse of covariance.
  arma::vec invCov;
  //! Cached logdet(cov).
  double logDetCov;

  //! log(2pi)
  static const constexpr double log2pi = 1.83787706640934533908193770912475883;

 public:
  /**
   * Default constructor, which creates a Gaussian with zero dimension.
   */
  DiagCovGaussianDistribution() : logDetCov(0.0) { /* nothing to do. */ }
  
  /**
   * Create a Gaussian Distribution with zero mean and diagonal covariance
   * with the given dimensionality.
   */
  DiagCovGaussianDistribution(const size_t dimension) :
      mean(arma::zeros<arma::vec>(dimension)),
      covariance(arma::eye<arma::mat>(dimension, dimension)),
      invCov(arma::ones<arma::vec>(dimension)),
      logDetCov(0)
  { /* nothing to do. */ }

  /**
   * Create a Guassian distribution with the given mean and diagonal
   * covariance.
   */
  DiagCovGaussianDistribution(const arma::vec& mean, 
                              const arma::mat& covariance);

  /**
   * Return the dimensionalty of this distribution.
   */
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
   * data point (column) in the given matrix.
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
  * Calculates the multivariate Gaussian log probability density function for
  * each data point (column) in the given matrix.
  *
  * @param observations Matrix of observations.
  * @param probabilities Output log probabilities for each input observation.
  */
  void LogProbability(const arma::mat& observations,
                      arma::vec& logProbabilites) const;

  /**
   * Caculates the log determinant of the given diagonal covariance.
   */
  void LogDeterminant();

  /**
   * Calculates the inverse of the given diagonal covariance.
   */
  void InverseCovariance();

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   * 
   * @ return Random observation from this Diagonal Gaussian distribution.
   */
  arma::vec Random() const;

  /**
   * Estimate the Gaussian distribution directly from the given observations.
   * 
   * @param observations List of observations.
   */
  void Train(const arma::mat& observations);

  /**
   * Estimate the Gaussian distribution from the given observations,
   * taking into account the probability of each observation actually being
   * from this distribution.
   * 
   * @param observations Matrix of observations.
   * @param probabilities List of probability of the each observation being
   * from this distribution.
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
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    // We just need to serialize each of the members.
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(covariance);
    ar & BOOST_SERIALIZATION_NVP(invCov);
    ar & BOOST_SERIALIZATION_NVP(logDetCov);
  }
};

/**
* Calculates the multivariate Gaussian log probability density function for
* each data point (column) in the given matrix.
*
* @param observations Matrix of observations.
* @param probabilities Output log probabilities for each input observation.
*/
inline void DiagCovGaussianDistribution::LogProbability(
    const arma::mat& observations,
    arma::vec& logProbabilities) const
{
  // Column i of 'diffs' is the difference between observations.col(i) and
  // the mean.
  arma::mat diffs = observations -
      (mean * arma::ones<arma::rowvec>(observations.n_cols));

  // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1 *
  // diffs).  We just don't need any of the other elements.  We can calculate
  // the right hand part of the equation (instead of the left side) so that
  // later we are referencing columns, not rows -- that is faster.
  const size_t k = observations.n_rows;
  const arma::mat rhs = -0.5 * arma::diagmat(invCov) * diffs;
  arma::vec logExponents(diffs.n_cols);
  
  for (size_t i = 0; i < diffs.n_cols; i++)
    logExponents(i) = accu(diffs.unsafe_col(i) % rhs.unsafe_col(i));

  logProbabilities = -0.5 * k * log2pi - 0.5 * logDetCov + logExponents;
}

} // namespace distribution
} // namespace mlpack

#endif
