/*
 * @file core/dists/laplace_distribution_impl.hpp
 * @author Zhihao Lou
 * @author Rohan Raj
 *
 * Implementation of Laplace distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_LAPLACE_DISTRIBUTION_IMPL_HPP
#define MLPACK_CORE_DISTRIBUTIONS_LAPLACE_DISTRIBUTION_IMPL_HPP

#include "laplace_distribution.hpp"

namespace mlpack {

/**
 * Return the log probability of the given observation.
 */
inline double LaplaceDistribution::LogProbability(
    const arma::vec& observation) const
{
  // Evaluate the PDF of the Laplace distribution to determine
  // the log probability.
  return -std::log(2. * scale) - arma::norm(observation - mean, 2) / scale;
}

/**
 * Evaluate probability density function of given observation.
 *
 * @param x List of observations.
 * @param probabilities Output probabilities for each input observation.
 */
inline void LaplaceDistribution::Probability(const arma::mat& x,
                                             arma::vec& probabilities) const
{
  probabilities.set_size(x.n_cols);
  for (size_t i = 0; i < x.n_cols; ++i)
  {
    probabilities(i) = Probability(x.unsafe_col(i));
  }
}

/**
 * Estimate the Laplace distribution directly from the given observations.
 *
 * @param observations List of observations.
 */
inline void LaplaceDistribution::Estimate(const arma::mat& observations)
{
  // The maximum likelihood estimate of the mean is the median of the data for
  // the univariate case.  See the short note "The Double Exponential
  // Distribution: Using Calculus to Find a Maximum Likelihood Estimator" by
  // R.M. Norton.
  //
  // But for the multivariate case, the derivation is slightly different.  The
  // log-likelihood function is now
  //   L(\theta) = -n ln 2 - \sum_{i = 1}^{n} (x_i - \theta)^T (x_i - \theta).
  // Differentiating with respect to the vector \theta gives
  //   L'(\theta) = \sum_{i = 1}^{n} 2 (x_i - \theta)
  // which means that for an individual component \theta_k,
  //   d / d\theta_k L(\theta) = \sum_{i = 1}^{n} 2 (x_ik - \theta_k)
  // which is zero when
  //   \theta_k = (1 / n) \sum_{i = 1}^{n} x_ik
  // so L'(\theta) = 0 when \theta is the mean of the observations.  I am not
  // 100% certain my calculus and linear algebra is right, but I think it is...
  mean = arma::mean(observations, 1);

  // The maximum likelihood estimate of the scale parameter is the mean
  // deviation from the mean.
  scale = 0.0;
  for (size_t i = 0; i < observations.n_cols; ++i)
    scale += arma::norm(observations.col(i) - mean, 2);
  scale /= observations.n_cols;
}

/**
 * Estimate the Laplace distribution directly from the given observations,
 * taking into account the probability of each observation actually being from
 * this distribution.
 */
inline void LaplaceDistribution::Estimate(const arma::mat& observations,
                                          const arma::vec& probabilities)
{
  // I am not completely sure that this change results in a valid maximum
  // likelihood estimator given probabilities of points.
  mean.zeros(observations.n_rows);
  for (size_t i = 0; i < observations.n_cols; ++i)
    mean += observations.col(i) * probabilities(i);
  mean /= accu(probabilities);

  // This is the same formula as the previous function, but here we are
  // multiplying by the probability that the point is actually from
  // this distribution.
  scale = 0.0;
  for (size_t i = 0; i < observations.n_cols; ++i)
    scale += probabilities(i) * arma::norm(observations.col(i) - mean, 2);
  scale /= accu(probabilities);
}

} // namespace mlpack

#endif
