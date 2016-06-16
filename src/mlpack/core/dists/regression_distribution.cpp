/**
 * @file regression_distribution.cpp
 * @author Michael Fox
 *
 * Implementation of conditional Gaussian distribution for HMM regression (HMMR)
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
