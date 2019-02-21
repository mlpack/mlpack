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
    const arma::vec& covariance) :
    mean(mean)
{
  Covariance(covariance);
}

void DiagCovGaussianDistribution::Covariance(const arma::vec& covariance)
{
  this->covariance = covariance;
  InvertCovariance();
  LogDeterminant();
}

void DiagCovGaussianDistribution::Covariance(arma::vec&& covariance)
{
  this->covariance = std::move(covariance);
  InvertCovariance();
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

void DiagCovGaussianDistribution::InvertCovariance()
{
  // Calculate the inverse of the diagonal covariance.
  invCov = 1/covariance;
}

void DiagCovGaussianDistribution::LogDeterminant()
{
  // Calculate Log determinant of the diagonal covariance.
  logDetCov = arma::accu(log(covariance));
}

arma::vec DiagCovGaussianDistribution::Random() const
{
  return (arma::sqrt(covariance) % arma::randn<arma::vec>(mean.n_elem)) + mean;
}

void DiagCovGaussianDistribution::Train(const arma::mat& observations)
{
  if (observations.n_cols > 1)
  {
    covariance.zeros(observations.n_rows);
  }
  else
  {
    mean.zeros(0);
    covariance.zeros(0);
    return;
  }

  // Calculate the mean.
  mean = arma::sum(observations, 1);

  // Normalize the mean.
  mean /= observations.n_cols;

  // Now calculate the covariance.
  const arma::mat diffs = observations - mean *
      arma::ones<arma::rowvec>(observations.n_cols);
  covariance += arma::sum(diffs % diffs, 1);

  // Finish estimating the covariance by normalizing, with the (1 / (n - 1))
  // to make the estimator unbiased.
  covariance /= (observations.n_cols - 1);

  InvertCovariance();
  LogDeterminant();
}

void DiagCovGaussianDistribution::Train(const arma::mat& observations,
                                        const arma::vec& probabilities)
{
  if (observations.n_cols > 0)
  {
    covariance.zeros(observations.n_rows);
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
  arma::vec normalizedProbs = probabilities;

  // If their sum is 0, there is nothing in this Gaussian.
  // At least, set the covariance so that it's invertible.
  if (v1 == 0)
  {
    covariance += 1e-50;
    InvertCovariance();
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
  mean = observations * normalizedProbs;

  // Now calculate the covariance.
  const arma::mat diffs = observations - mean *
      arma::ones<arma::rowvec>(observations.n_cols);
  covariance += (diffs % diffs) * normalizedProbs;

  // Calculate the sum of each weight squared.
  const double v2 = arma::accu(normalizedProbs % normalizedProbs);

  // Finish estimating the covariance by normalizing, with
  // the (1 / (v1 - (v2 / v1))) to make the estimator unbiased.
  if (v2 != 1)
    covariance /= (1 - v2);

  InvertCovariance();
  LogDeterminant();
}
