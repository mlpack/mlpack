/**
 * @file gaussian_distribution.cpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Implementation of Gaussian distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "gaussian_distribution.hpp"
#include <mlpack/methods/gmm/positive_definite_constraint.hpp>

using namespace mlpack;
using namespace mlpack::distribution;


GaussianDistribution::GaussianDistribution(const arma::vec& mean,
                                           const arma::mat& covariance)
  : mean(mean)
{
  Covariance(covariance);
}

void GaussianDistribution::Covariance(const arma::mat& covariance)
{
  this->covariance = covariance;
  FactorCovariance();
}

void GaussianDistribution::Covariance(arma::mat&& covariance)
{
  this->covariance = std::move(covariance);
  FactorCovariance();
}

void GaussianDistribution::FactorCovariance()
{
  // On Armadillo < 4.500, the "lower" option isn't available.
  #if (ARMA_VERSION_MAJOR < 4) || \
      ((ARMA_VERSION_MAJOR == 4) && (ARMA_VERSION_MINOR < 500))
    covLower = arma::chol(covariance).t(); // This is less efficient.
  #else
    covLower = arma::chol(covariance, "lower");
  #endif

  // Comment from rcurtin:
  //
  // I think the use of the word "interpret" in the Armadillo documentation
  // about trimatl and trimatu is somewhat misleading. What the function will
  // actually do, when used in that context, is loop over the upper triangular
  // part of the matrix and set it all to 0, so this ends up actually just
  // burning cycles---also because the operator=() evaluates the expression and
  // strips the knowledge that it's a lower triangular matrix. So then the call
  // to .i() doesn't actually do anything smarter.
  //
  // But perusing fn_inv.hpp more closely, there is a specialization that will
  // work when called like this: inv(trimatl(covLower)), and will use LAPACK's
  // ?trtri functions. However, it will still set the upper triangular part to
  // 0 after the method. That last part is unnecessary, but baked into
  // Armadillo, so there's not really much that can be done about that without
  // discussion with the Armadillo maintainer.
  const arma::mat invCovLower = arma::inv(arma::trimatl(covLower));

  invCov = invCovLower.t() * invCovLower;
  double sign = 0.;
  arma::log_det(logDetCov, sign, covLower);
  logDetCov *= 2;
}

double GaussianDistribution::LogProbability(const arma::vec& observation) const
{
  const size_t k = observation.n_elem;
  const arma::vec diff = mean - observation;
  const arma::vec v = (diff.t() * invCov * diff);
  return -0.5 * k * log2pi - 0.5 * logDetCov - 0.5 * v(0);
}

arma::vec GaussianDistribution::Random() const
{
  return covLower * arma::randn<arma::vec>(mean.n_elem) + mean;
}

/**
 * Estimate the Gaussian distribution directly from the given observations.
 *
 * @param observations List of observations.
 */
void GaussianDistribution::Train(const arma::mat& observations)
{
  if (observations.n_cols > 0)
  {
    mean.zeros(observations.n_rows);
    covariance.zeros(observations.n_rows, observations.n_rows);
  }
  else // This will end up just being empty.
  {
    // TODO(stephentu): why do we allow this case? why not throw an error?
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
    covariance += obsNoMean * trans(obsNoMean);
  }

  // Finish estimating the covariance by normalizing, with the (1 / (n - 1)) so
  // that it is the unbiased estimator.
  covariance /= (observations.n_cols - 1);

  // Ensure that the covariance is positive definite.
  gmm::PositiveDefiniteConstraint::ApplyConstraint(covariance);

  FactorCovariance();
}

/**
 * Estimate the Gaussian distribution from the given observations, taking into
 * account the probability of each observation actually being from this
 * distribution.
 */
void GaussianDistribution::Train(const arma::mat& observations,
                                 const arma::vec& probabilities)
{
  if (observations.n_cols > 0)
  {
    mean.zeros(observations.n_rows);
    covariance.zeros(observations.n_rows, observations.n_rows);
  }
  else // This will end up just being empty.
  {
    // TODO(stephentu): same as above
    mean.zeros(0);
    covariance.zeros(0);
    return;
  }

  double sumProb = 0;

  // First calculate the mean, and save the sum of all the probabilities for
  // later normalization.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    mean += probabilities[i] * observations.col(i);
    sumProb += probabilities[i];
  }

  if (sumProb == 0)
  {
    // Nothing in this Gaussian!  At least set the covariance so that it's
    // invertible.
    covariance.diag() += 1e-50;
    FactorCovariance();
    return;
  }

  // Normalize.
  if (sumProb > 0)
    mean /= sumProb;

  // Now find the covariance.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    arma::vec obsNoMean = observations.col(i) - mean;
    covariance += probabilities[i] * (obsNoMean * trans(obsNoMean));
  }

  // This is probably biased, but I don't know how to unbias it.
  if (sumProb > 0)
    covariance /= sumProb;

  // Ensure that the covariance is positive definite.
  gmm::PositiveDefiniteConstraint::ApplyConstraint(covariance);

  FactorCovariance();
}
