/**
 * @file core/dists/regression_distribution_impl.hpp
 * @author Michael Fox
 *
 * Implementation of conditional Gaussian distribution for HMM regression
 * (HMMR).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTS_REGRESSION_DISTRIBUTION_IMPL_HPP
#define MLPACK_CORE_DISTS_REGRESSION_DISTRIBUTION_IMPL_HPP

#include "regression_distribution.hpp"

namespace mlpack {

/**
 * Estimate parameters using provided observation weights.
 *
 * @param observations List of observations.
 */
inline void RegressionDistribution::Train(const arma::mat& observations)
{
  LinearRegression<> lr(observations.rows(1, observations.n_rows - 1),
      arma::rowvec(observations.row(0)), 0, true);
  rf = lr;
  arma::rowvec fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Train(observations.row(0) - fitted);
}

/**
 * Estimate parameters using provided observation weights.
 *
 * @param weights Probability that given observation is from distribution.
 */
inline void RegressionDistribution::Train(const arma::mat& observations,
                                          const arma::vec& weights)
{
  Train(observations, arma::rowvec(weights.t()));
}

inline void RegressionDistribution::Train(const arma::mat& observations,
                                          const arma::rowvec& weights)
{
  LinearRegression<> lr(observations.rows(1, observations.n_rows - 1),
      arma::rowvec(observations.row(0)), weights, 0, true);
  rf = lr;
  arma::rowvec fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Train(observations.row(0) - fitted, weights.t());
}

/**
 * Evaluate probability density function of given observation.
 *
 * @param observation Point to evaluate probability at.
 */
inline double RegressionDistribution::Probability(
    const arma::vec& observation) const
{
  arma::rowvec fitted;
  rf.Predict(observation.rows(1, observation.n_rows - 1), fitted);
  return err.Probability(observation(0) - fitted.t());
}

inline void RegressionDistribution::Predict(const arma::mat& points,
                                            arma::vec& predictions) const
{
  arma::rowvec rowPredictions;
  Predict(points, rowPredictions);
  predictions = rowPredictions.t();
}

inline void RegressionDistribution::Predict(const arma::mat& points,
                                            arma::rowvec& predictions) const
{
  rf.Predict(points, predictions);
}

} // namespace mlpack

#endif
