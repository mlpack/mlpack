/**
 * @file core/distributions/diagonal_gaussian_distribution.hpp
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

//! A single multivariate Gaussian distribution with diagonal covariance.
template<typename MatType = arma::mat>
class DiagonalGaussianDistribution
{
 public:
  // Convenience typedefs.
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;

 private:
  //! Mean of the distribution.
  VecType mean;
  //! Diagonal covariance of the distribution.
  VecType covariance;
  //! Cached inverse of covariance.
  VecType invCov;
  //! Cached logdet(cov).
  ElemType logDetCov;

  //! log(2pi)
  static const constexpr ElemType log2pi =
      1.83787706640934533908193770912475883;

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
      mean(arma::zeros<VecType>(dimension)),
      covariance(arma::ones<VecType>(dimension)),
      invCov(arma::ones<VecType>(dimension)),
      logDetCov(0)
  { /* Nothing to do. */ }

  /**
   * Create a Gaussian distribution with the given mean and diagonal
   * covariance.
   *
   * @param mean Mean of distribution.
   * @param covariance Covariance of distribution.
   */
  DiagonalGaussianDistribution(const VecType& mean,
                               const VecType& covariance);

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  //! Return the probability of the given observation.
  ElemType Probability(const VecType& observation) const
  {
    return std::exp(LogProbability(observation));
  }

  //! Return the log probability of the given observation.
  ElemType LogProbability(const VecType& observation) const;

  /**
   * Calculate the multivariate Gaussian probability density function for each
   * data point (column) in the given matrix.
   *
   * @param x Matrix of observations.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const MatType& x, VecType& probabilities) const
  {
    VecType logProbabilities;
    LogProbability(x, logProbabilities);
    probabilities = exp(logProbabilities);
  }

  /**
   * Calculate the multivariate Gaussian log probability density function for
   * each data point (column) in the given matrix.
   *
   * @param observations Matrix of observations.
   * @param logProbabilities Output log probabilities for each observation.
   */
  void LogProbability(const MatType& observations,
                      VecType& logProbabilities) const;

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this Diagonal Gaussian distribution.
   */
  VecType Random() const;

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
             const VecType& probabilities);

  //! Return the mean.
  const VecType& Mean() const { return mean; }

  //! Return a modifiable copy of the mean.
  VecType& Mean() { return mean; }

  //! Return the covariance matrix.
  const VecType& Covariance() const { return covariance; }

  //! Set the covariance matrix.
  void Covariance(const VecType& covariance);

  //! Set the covariance matrix using move assignment.
  void Covariance(VecType&& covariance);

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

} // namespace mlpack

// Include implementation.
#include "diagonal_gaussian_distribution_impl.hpp"

#endif
