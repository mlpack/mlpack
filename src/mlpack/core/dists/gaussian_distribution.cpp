/**
 * @file gaussian_distribution.cpp
 * @author Ryan Curtin
 *
 * Implementation of Gaussian distribution class.
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
void GaussianDistribution::Estimate(const arma::mat& observations)
{
  if (observations.n_cols > 0)
  {
    mean.zeros(observations.n_rows);
    covariance.zeros(observations.n_rows, observations.n_rows);
  }
  else // This will end up just being empty.
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
    covariance += obsNoMean * trans(obsNoMean);
  }

  // Finish estimating the covariance by normalizing, with the (1 / (n - 1)) so
  // that it is the unbiased estimator.
  covariance /= (observations.n_cols - 1);

  // Ensure that the covariance is positive definite.
  if (det(covariance) <= 1e-50)
  {
    Log::Debug << "GaussianDistribution::Estimate(): Covariance matrix is not "
        << "positive definite. Adding perturbation." << std::endl;

    double perturbation = 1e-30;
    while (det(covariance) <= 1e-50)
    {
      covariance.diag() += perturbation;
      perturbation *= 10; // Slow, but we don't want to add too much.
    }
  }
}

/**
 * Estimate the Gaussian distribution from the given observations, taking into
 * account the probability of each observation actually being from this
 * distribution.
 */
void GaussianDistribution::Estimate(const arma::mat& observations,
                                    const arma::vec& probabilities)
{
  if (observations.n_cols > 0)
  {
    mean.zeros(observations.n_rows);
    covariance.zeros(observations.n_rows, observations.n_rows);
  }
  else // This will end up just being empty.
  {
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
    return;
  }

  // Normalize.
  mean /= sumProb;

  // Now find the covariance.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    arma::vec obsNoMean = observations.col(i) - mean;
    covariance += probabilities[i] * (obsNoMean * trans(obsNoMean));
  }

  // This is probably biased, but I don't know how to unbias it.
  covariance /= sumProb;

  // Ensure that the covariance is positive definite.
  if (det(covariance) <= 1e-50)
  {
    Log::Debug << "GaussianDistribution::Estimate(): Covariance matrix is not "
        << "positive definite. Adding perturbation." << std::endl;

    double perturbation = 1e-30;
    while (det(covariance) <= 1e-50)
    {
      covariance.diag() += perturbation;
      perturbation *= 10; // Slow, but we don't want to add too much.
    }
  }
}

/**
 * Returns a string representation of this object.
 */
std::string GaussianDistribution::ToString() const
{
  std::ostringstream convert;
  convert << "GaussianDistribution [" << this << "]" << std::endl;

  // Secondary ostringstream so things can be indented right.
  std::ostringstream data;
  data << "Mean: " << std::endl << mean;
  data << "Covariance: " << std::endl << covariance;

  convert << util::Indent(data.str());
  return convert.str();
}
