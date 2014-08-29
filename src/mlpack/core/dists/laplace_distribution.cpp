/*
 * @file laplace_distribution.cpp
 * @author Zhihao Lou
 *
 * Implementation of Laplace distribution.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include "laplace_distribution.hpp"

using namespace mlpack;
using namespace mlpack::distribution;

/**
 * Return the probability of the given observation.
 */
double LaplaceDistribution::Probability(const arma::vec& observation) const
{
  // Evaluate the PDF of the Laplace distribution to determine the probability.
  return (0.5 / scale) * std::exp(arma::norm(observation - mean, 2) / scale);
}

/**
 * Estimate the Laplace distribution directly from the given observations.
 *
 * @param observations List of observations.
 */
void LaplaceDistribution::Estimate(const arma::mat& observations)
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
void LaplaceDistribution::Estimate(const arma::mat& observations,
                                   const arma::vec& probabilities)
{
  // I am not completely sure that this change results in a valid maximum
  // likelihood estimator given probabilities of points.
  mean.zeros(observations.n_rows);
  for (size_t i = 0; i < observations.n_cols; ++i)
    mean += observations.col(i) * probabilities(i);
  mean /= arma::accu(probabilities);

  // This the same formula as the previous function, but here we are multiplying
  // by the probability that the point is actually from this distribution.
  scale = 0.0;
  for (size_t i = 0; i < observations.n_cols; ++i)
    scale += probabilities(i) * arma::norm(observations.col(i) - mean, 2);
  scale /= arma::accu(probabilities);
}

//! Returns a string representation of this object.
std::string LaplaceDistribution::ToString() const
{
  std::ostringstream convert;
  convert << "LaplaceDistribution [" << this << "]" << std::endl;

  std::ostringstream data;
  data << "Mean: " << std::endl << mean.t();
  data << "Scale: " << scale << "." << std::endl;

  convert << util::Indent(data.str());
  return convert.str();
}
