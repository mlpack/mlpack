/**
 * @file core/dists/diagonal_gaussian_distribution.hpp
 * @author Kim SangYeon
 *
 * Implementation of the Gaussian distribution with diagonal covariance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_DIAGONAL_GAUSSIAN_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_DIAGONAL_GAUSSIAN_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace distribution {

//! A single multivariate Gaussian distribution with diagonal covariance.
template<typename MatType, typename Vectype>
class DiagonalGaussianDistribution
{
 private:
  //! Mean of the distribution.
  Vectype mean;
  //! Diagonal covariance of the distribution.
  Vectype covariance;
  //! Cached inverse of covariance.
  Vectype invCov;
  //! Cached logdet(cov).
  double logDetCov;

  //! log(2pi)
  static const constexpr double log2pi = 1.83787706640934533908193770912475883;

 public:
  //! Default constructor, which creates a Gaussian with zero dimension.
  DiagonalGaussianDistribution() : logDetCov(0.0) { /* nothing to do. */ }

  /**
   * Create a Gaussian Distribution with zero mean and diagonal covariance
   * with the given dimensionality.
   *
   * @param dimension Number of dimensions.
   */
  DiagonalGaussianDistribution(const size_t dimension) :
      mean(arma::zeros<Vectype>(dimension)),
      covariance(arma::ones<Vectype>(dimension)),
      invCov(arma::ones<Vectype>(dimension)),
      logDetCov(0)
  { /* Nothing to do. */ }

  /**
   * Create a Gaussian distribution with the given mean and diagonal
   * covariance.
   *
   * @param mean Mean of distribution.
   * @param covariance Covariance of distribution.
   */
  DiagonalGaussianDistribution(const Vectype& mean,
                               const Vectype& covariance);

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  //! Return the probability of the given observation.
  double Probability(const Vectype& observation) const
  {
    return exp(LogProbability(observation));
  }

  //! Return the log probability of the given observation.
  double LogProbability(const Vectype& observation) const;

  /**
   * Calculate the multivariate Gaussian probability density function for each
   * data point (column) in the given matrix.
   *
   * @param x Matrix of observations.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const MatType& x, Vectype& probabilities) const
  {
    Vectype logProbabilities;
    LogProbability(x, logProbabilities);
    probabilities = arma::exp(logProbabilities);
  }

  /**
   * Calculate the multivariate Gaussian log probability density function for
   * each data point (column) in the given matrix.
   *
   * @param observations Matrix of observations.
   * @param logProbabilities Output log probabilities for each observation.
   */
  void LogProbability(const MatType& observations,
                      Vectype& logProbabilities) const;

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this Diagonal Gaussian distribution.
   */
  Vectype Random() const;

  /**
   * Estimate the Gaussian distribution directly from the given observations.
   *
   * @param observations Matrix of observations.
   */
  void Train(const MatType& observations);

  /**
   * Estimate the Gaussian distribution from the given observations,
   * taking into account the probability of each observation actually being
   * from this distribution.
   *
   * @param observations Matrix of observations.
   * @param probabilities List of probability of the each observation being
   * from this distribution.
   */
  void Train(const MatType& observations,
             const Vectype& probabilities);

  //! Return the mean.
  const Vectype& Mean() const { return mean; }

  //! Return a modifiable copy of the mean.
  Vectype& Mean() { return mean; }

  //! Return the covariance matrix.
  const Vectype& Covariance() const { return covariance; }

  //! Set the covariance matrix.
  void Covariance(const Vectype& covariance);

  //! Set the covariance matrix using move assignment.
  void Covariance(Vectype&& covariance);

  //! Serialize the distribution.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    // We just need to serialize each of the members.
    ar(CEREAL_NVP(mean));
    ar(CEREAL_NVP(covariance));
    ar(CEREAL_NVP(invCov));
    ar(CEREAL_NVP(logDetCov));
  }
};

} // namespace distribution
} // namespace mlpack

#include "diagonal_gaussian_distribution_impl.hpp"

#endif
