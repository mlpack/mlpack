/**
 * @file gaussian_distribution.hpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Implementation of the Gaussian distribution.
 */
#ifndef __MLPACK_METHODS_HMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP
#define __MLPACK_METHODS_HMM_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP

#include <mlpack/core.hpp>

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
  double Probability(const arma::vec& observation) const;
  
  /**
   * Calculates the multivariate Gaussian probability density function for each
   * data point (column) in the given matrix
   *
   * @param x List of observations.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const arma::mat& x, arma::vec& probabilities) const;
  
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

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;
    
  /*
   * Save to or Load from SaveRestoreUtility
   */
  void Save(util::SaveRestoreUtility& n) const;
  void Load(const util::SaveRestoreUtility& n);
  static std::string const Type() { return "GaussianDistribution"; }
  
  
    
};

/**
* Calculates the multivariate Gaussian probability density function for each
* data point (column) in the given matrix
*
* @param x List of observations.
* @param probabilities Output probabilities for each input observation.
*/
inline void GaussianDistribution::Probability(const arma::mat& x,
                                              arma::vec& probabilities) const
{
  // Column i of 'diffs' is the difference between x.col(i) and the mean.
  arma::mat diffs = x - (mean * arma::ones<arma::rowvec>(x.n_cols));
  
  // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1 *
  // diffs).  We just don't need any of the other elements.  We can calculate
  // the right hand part of the equation (instead of the left side) so that
  // later we are referencing columns, not rows -- that is faster.
  arma::mat rhs = -0.5 * inv(covariance) * diffs;
  arma::vec exponents(diffs.n_cols); // We will now fill this.
  for (size_t i = 0; i < diffs.n_cols; i++)
    exponents(i) = exp(accu(diffs.unsafe_col(i) % rhs.unsafe_col(i)));
  
  probabilities = pow(2 * M_PI, (double) mean.n_elem / -2.0) *
  pow(arma::det(covariance), -0.5) * exponents;
}
  

}; // namespace distribution
}; // namespace mlpack

#endif
