/**
 * @file core/distributions/gaussian_distribution.hpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Implementation of the Gaussian distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A single multivariate Gaussian distribution.
 */
template<typename MatType = arma::mat>
class GaussianDistribution
{
 public:
  // Convenience typedefs for derived types of MatType.
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;

 private:
  //! Mean of the distribution.
  VecType mean;
  //! Positive definite covariance of the distribution.
  MatType covariance;
  //! Lower triangular factor of cov (e.g. cov = LL^T).
  MatType covLower;
  //! Cached inverse of covariance.
  MatType invCov;
  //! Cached logdet(cov).
  ElemType logDetCov;

  //! log(2pi)
  static const constexpr ElemType log2pi =
      1.83787706640934533908193770912475883;

 public:
  /**
   * Default constructor, which creates a Gaussian with zero dimension.
   */
  GaussianDistribution() : logDetCov(0.0) { /* nothing to do */ }

  /**
   * Create a Gaussian distribution with zero mean and identity covariance with
   * the given dimensionality.
   */
  GaussianDistribution(const size_t dimension) :
      mean(arma::zeros<VecType>(dimension)),
      covariance(arma::eye<MatType>(dimension, dimension)),
      covLower(arma::eye<MatType>(dimension, dimension)),
      invCov(arma::eye<MatType>(dimension, dimension)),
      logDetCov(0)
  { /* Nothing to do. */ }

  /**
   * Create a Gaussian distribution with the given mean and covariance.
   *
   * covariance is expected to be positive definite.
   */
  GaussianDistribution(const VecType& mean, const MatType& covariance);

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  /**
   * Return the probability of the given observation.
   */
  ElemType Probability(const VecType& observation) const
  {
    return std::exp(LogProbability(observation));
  }

  /**
   * Return the log probability of the given observation.
   */
  ElemType LogProbability(const VecType& observation) const;

  /**
   * Calculates the multivariate Gaussian probability density function for each
   * data point (column) in the given matrix.
   *
   * @param x List of observations.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const MatType& x, VecType& probabilities) const
  {
    // Use LogProbability(), then transform the log-probabilities out of
    // logspace.
    VecType logProbs;
    LogProbability(x, logProbs);
    probabilities = exp(logProbs);
  }

  /**
   * Returns the Log probability of the given matrix. These values are stored
   * in logProbabilities.
   *
   * @param x List of observations.
   * @param logProbabilities Output log probabilities for each input
   *     observation.
   */
  void LogProbability(const MatType& x, VecType& logProbabilities) const
  {
    // Column i of 'diffs' is the difference between x.col(i) and the mean.
    MatType diffs = x;
    diffs.each_col() -= mean;

    // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1
    // * diffs).  We just don't need any of the other elements.
    logProbabilities = -0.5 * x.n_rows * log2pi - 0.5 * logDetCov +
        sum(diffs % (-0.5 * invCov * diffs), 0).t();
  }

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this Gaussian distribution.
   */
  VecType Random() const;

  /**
   * Estimate the Gaussian distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Train(const MatType& observations);

  /**
   * Estimate the Gaussian distribution from the given observations, taking
   * into account the probability of each observation actually being from this
   * distribution.
   */
  void Train(const MatType& observations,
             const VecType& probabilities);

  /**
   * Return the mean.
   */
  const VecType& Mean() const { return mean; }

  /**
   * Return a modifiable copy of the mean.
   */
  VecType& Mean() { return mean; }

  /**
   * Return the covariance matrix.
   */
  const MatType& Covariance() const { return covariance; }

  /**
   * Set the covariance.
   */
  void Covariance(const MatType& covariance);
  void Covariance(MatType&& covariance);

  //! Return the invCov.
  const MatType& InvCov() const { return invCov; }

  //! Return the logDetCov.
  ElemType LogDetCov() const { return logDetCov; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    // We just need to serialize each of the members.
    ar(CEREAL_NVP(mean));
    ar(CEREAL_NVP(covariance));
    ar(CEREAL_NVP(covLower));
    ar(CEREAL_NVP(invCov));
    ar(CEREAL_NVP(logDetCov));
  }

 private:
  /**
   * This factors the covariance using arma::chol().  The function assumes that
   * the given matrix is factorizable via the Cholesky decomposition.  If not,
   * a std::runtime_error will be thrown.
   */
  void FactorCovariance();
};

} // namespace mlpack

// Include implementation.
#include "gaussian_distribution_impl.hpp"

#endif
