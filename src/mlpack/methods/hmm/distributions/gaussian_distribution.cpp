/**
 * @file gaussian_distribution.cpp
 * @author Ryan Curtin
 *
 * Implementation of Gaussian distribution class.
 */
#include "gaussian_distribution.hpp"

using namespace mlpack;
using namespace mlpack::distribution;

arma::vec GaussianDistribution::Random() const
{
  // Should we store chol(covariance) for easier calculation later?
  return trans(chol(covariance)) * arma::randn<arma::vec>(mean.n_elem) + mean;
}

/**
 * Estimate the Gaussian distribution directly from the given observations.
 *
 * @param observations List of observations.
 */
void GaussianDistribution::Estimate(const std::vector<arma::vec> observations)
{
  // Calculate the mean and covariance with each point.  Because this is a
  // std::vector and not a matrix, this is a little more difficult.
  if (observations.size() > 0)
  {
    mean.zeros(observations[0].n_elem);
    covariance.zeros(observations[0].n_elem, observations[0].n_elem);
  }
  else // This will end up just being empty.
  {
    mean.zeros(0);
    covariance.zeros(0);
  }

  // Calculate the mean.
  for (size_t i = 0; i < observations.size(); i++)
    mean += observations[i];

  // Normalize the mean.
  mean /= observations.size();

  // Now calculate the covariance.
  for (size_t i = 0; i < observations.size(); i++)
  {
    arma::vec obsNoMean = observations[i] - mean;
    covariance += obsNoMean * trans(obsNoMean);
  }

  // Finish estimating the covariance by normalizing, with the (1 / (n - 1)) so
  // that it is the unbiased estimator.
  covariance /= (observations.size() - 1);
}

/**
 * Estimate the Gaussian distribution from the given observations, taking into
 * account the probability of each observation actually being from this
 * distribution.
 */
void GaussianDistribution::Estimate(const std::vector<arma::vec> observations,
                                    const std::vector<double> probabilities)
{
  if (observations.size() > 0)
  {
    mean.zeros(observations[0].n_elem);
    covariance.zeros(observations[0].n_elem, observations[0].n_elem);
  }
  else // This will end up just being empty.
  {
    mean.zeros(0);
    covariance.zeros(0);
  }

  double sumProb = 0;

  // First calculate the mean, and save the sum of all the probabilities for
  // later normalization.
  for (size_t i = 0; i < observations.size(); i++)
  {
    mean += probabilities[i] * observations[i];
    sumProb += probabilities[i];
  }

  // Normalize.
  mean /= sumProb;

  // Now find the covariance.
  for (size_t i = 0; i < observations.size(); i++)
  {
    arma::vec obsNoMean = observations[i] - mean;
    covariance += probabilities[i] * (obsNoMean * trans(obsNoMean));
  }

  // This is probably biased, but I don't know how to unbias it.
  covariance /= sumProb;
}
