/**
 * @file diag_cov_gaussian_distribution.cpp
 * @author Kim SangYeon
 *
 * Implementation of Gaussian distribution class with diagonal covariance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "diag_cov_gaussian_distribution.hpp"
#include <mlpack/methods/gmm/diagonal_constraint.hpp>

using namespace mlpack;
using namespace mlpack::distribution;

DiagCovGaussianDistribution::DiagCovGaussianDistribution(
    const arma::vec& mean,
    const arma::mat& covariance)
  : mean(mean)
{
  Covariance(covariance);
}

void DiagCovGaussianDistribution::Covariance(const arma::mat& covariance)
{
  this->covariance = arma::diagmat(covariance);
  InverseCovariance();
  LogDeterminant();
}

void DiagCovGaussianDistribution::Covariance(arma::mat&& covariance)
{
  this->covariance = std::move(covariance);
  this->covariance = arma::diagmat(this->covariance);
  InverseCovariance();
  LogDeterminant();
}

double DiagCovGaussianDistribution::LogProbability(
    const arma::vec& observation) const
{
  const size_t k = observation.n_elem;
  const arma::vec diff = observation - mean;
  const arma::vec logExponent = (diff.t() * arma::diagmat(invCov) * diff);
  return -0.5 * k * log2pi - 0.5 * logDetCov - 0.5 * logExponent(0);
}

void DiagCovGaussianDistribution::InverseCovariance()
{
  // Calculate the inverse of the diagonal covariance.
  invCov = 1/covariance.diag();
}

void DiagCovGaussianDistribution::LogDeterminant()
{
  // Calculate Log determinant of the diagonal covariance.
  logDetCov = arma::accu(log(covariance.diag()));
}

arma::vec DiagCovGaussianDistribution::Random() const
{
  return arma::sqrt(covariance) * arma::randn<arma::vec>(mean.n_elem) + mean;
}

void DiagCovGaussianDistribution::Train(const arma::mat& observations)
{
  if (observations.n_cols > 1)
  {
    mean.zeros(observations.n_rows);
    covariance.zeros(observations.n_rows, observations.n_rows);
  }
  else
  {
    mean.zeros(0);
    covariance.zeros(0);
    return;
  }

  // Calculate the mean.
  for (size_t i = 0; i < observations.n_cols; i++)
    mean += observations.col(i);

  // Normalize the mean.
  mean /= observations.n_cols;

  // Now calculate the covariance.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    arma::vec obsNoMean = observations.col(i) - mean;
    covariance.diag() += arma::diagvec(obsNoMean * trans(obsNoMean));
  }

  // Finish estimating the covariance by normalizing, with the (1 / (n - 1))
  // to make the estimator unbiased.
  covariance.diag() /= (observations.n_cols - 1);

  // Ensure that the covariance is diagonal.
  gmm::DiagonalConstraint::ApplyConstraint(covariance);

  InverseCovariance();
  LogDeterminant();
}

void DiagCovGaussianDistribution::Train(const arma::mat& observations,
                                        const arma::vec& probabilities)
{
  if (observations.n_cols > 0)
  {
    mean.zeros(observations.n_rows);
    covariance.zeros(observations.n_rows, observations.n_rows);
  }
  else
  {
    mean.zeros(0);
    covariance.zeros(0);
    return;
  }

  // We'll normalize the covariance with (v1 - (v2 / v1))
  // for unbiased estimator in the weighted arithmetic mean. The v1 is the sum
  // of the weights, and the v2 is the sum of the each weight squared.
  // If you want to know more detailed description, 
  // please refer to https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
  double v1 = arma::accu(probabilities);
  double v2 = 0;
  arma::vec normalizedProbs = probabilities;

  // If their sum is 0, there is nothing in this Gaussian.
  // At least, set the covariance so that it's invertible.
  if (v1 == 0)
  {
    covariance.diag() += 1e-50;
    InverseCovariance();
    LogDeterminant();
    return;
  }

  // If their sum is not 1, divide the weights by them to normalize.
  if (v1 != 1)
  {
    normalizedProbs = probabilities / v1;
    v1 = arma::accu(normalizedProbs);
  }

  // Calculate the mean.
  for (size_t i = 0; i < observations.n_cols; i++)
    mean += normalizedProbs[i] * observations.col(i);

  // Now find the covariance.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    arma::vec obsNoMean = observations.col(i) - mean;
    covariance.diag() += arma::diagvec(
        normalizedProbs[i] * (obsNoMean * trans(obsNoMean)));
    v2 += normalizedProbs(i) * normalizedProbs(i);
  }

  // Finish estimating the covariance by normalizing, with
  // the (1 / (v1 - (v2 / v1))) to make the estimator unbiased.
  if (v2 != 1)
    covariance.diag() /= (1 - v2);

  // Ensure that the covariance is positive definite.
  gmm::DiagonalConstraint::ApplyConstraint(covariance);

  InverseCovariance();
  LogDeterminant();
}
