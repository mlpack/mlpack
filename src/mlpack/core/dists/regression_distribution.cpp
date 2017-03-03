/**
 * @file regression_distribution.cpp
 * @author Michael Fox
 *
 * Implementation of conditional Gaussian distribution for HMM regression (HMMR)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "regression_distribution.hpp"

using namespace mlpack;
using namespace mlpack::distribution;

/**
 * Estimate parameters using provided observation weights
 *
 * @param observations List of observations.
 */
void RegressionDistribution::Train(const arma::mat& observations)
{
  regression::LinearRegression lr(observations.rows(1, observations.n_rows - 1),
      (observations.row(0)).t(), 0, true);
  rf = lr;
  arma::vec fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Train(observations.row(0) - fitted.t());
}

/**
 * Estimate parameters using provided observation weights.
 *
 * @param weights probability that given observation is from distribution
 */
void RegressionDistribution::Train(const arma::mat& observations,
                                   const arma::vec& weights)
{
  regression::LinearRegression lr(observations.rows(1, observations.n_rows - 1),
      (observations.row(0)).t(), 0, true, weights);
  rf = lr;
  arma::vec fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Train(observations.row(0) - fitted.t(), weights);
}

/**
 * Evaluate probability density function of given observation.
 *
 * @param observation point to evaluate probability at
 */
double RegressionDistribution::Probability(const arma::vec& observation) const
{
  arma::vec fitted;
  rf.Predict(observation.rows(1, observation.n_rows-1), fitted);
  return err.Probability(observation(0)-fitted);
}

void RegressionDistribution::Predict(const arma::mat& points,
                                     arma::vec& predictions) const
{
  rf.Predict(points, predictions);
}
