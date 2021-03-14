/**
 * @file methods/gmm/gmm.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Implementation of template-based GMM methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "gmm.hpp"
#include <mlpack/core/math/log_add.hpp>

namespace mlpack {
namespace gmm {

/**
 * Create a GMM with the given number of Gaussians, each of which have the
 * specified dimensionality.  The means and covariances will be set to 0.
 *
 * @param gaussians Number of Gaussians in this GMM.
 * @param dimensionality Dimensionality of each Gaussian.
 */
GMM::GMM(const size_t gaussians, const size_t dimensionality) :
    gaussians(gaussians),
    dimensionality(dimensionality),
    dists(gaussians, distribution::GaussianDistribution(dimensionality)),
    weights(gaussians)
{
  // Set equal weights.  Technically this model is still valid, but only barely.
  weights.fill(1.0 / gaussians);
}

// Copy constructor for when the other GMM uses the same fitting type.
GMM::GMM(const GMM& other) :
    gaussians(other.Gaussians()),
    dimensionality(other.dimensionality),
    dists(other.dists),
    weights(other.weights) { /* Nothing to do. */ }

GMM& GMM::operator=(const GMM& other)
{
  gaussians = other.gaussians;
  dimensionality = other.dimensionality;
  dists = other.dists;
  weights = other.weights;

  return *this;
}

/**
 * Return the log probability of the given observation being from this GMM.
 */
double GMM::LogProbability(const arma::vec& observation) const
{
  // Sum the probability for each Gaussian in our mixture (and we have to
  // multiply by the prior for each Gaussian too).
  double sum = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < gaussians; ++i)
    sum = math::LogAdd(sum, log(weights[i]) +
        dists[i].LogProbability(observation));

  return sum;
}

/**
 * Return the probability of the given observation being from this GMM.
 */
double GMM::Probability(const arma::vec& observation) const
{
  return exp(LogProbability(observation));
}

/**
 * Return the log probability of the given observation being from the given
 * component in the mixture.
 */
double GMM::LogProbability(const arma::vec& observation,
                        const size_t component) const
{
  // We are only considering one Gaussian component -- so we only need to call
  // Probability() once.  We do consider the prior probability!
  return log(weights[component]) + dists[component].LogProbability(observation);
}

/**
 * Return the probability of the given observation being from the given
 * component in the mixture.
 */
double GMM::Probability(const arma::vec& observation,
                        const size_t component) const
{
  return exp(LogProbability(observation, component));
}

/**
 * Return a randomly generated observation according to the probability
 * distribution defined by this object.
 */
arma::vec GMM::Random() const
{
  // Determine which Gaussian it will be coming from.
  double gaussRand = math::Random();
  size_t gaussian = 0;

  double sumProb = 0;
  for (size_t g = 0; g < gaussians; g++)
  {
    sumProb += weights(g);
    if (gaussRand <= sumProb)
    {
      gaussian = g;
      break;
    }
  }

  arma::mat cholDecomp;
  if (!arma::chol(cholDecomp, dists[gaussian].Covariance()))
  {
    Log::Fatal << "Cholesky decomposition failed." << std::endl;
  }
  return trans(cholDecomp) *
      arma::randn<arma::vec>(dimensionality) + dists[gaussian].Mean();
}

/**
 * Classify the given observations as being from an individual component in this
 * GMM.
 */
void GMM::Classify(const arma::mat& observations,
                   arma::Row<size_t>& labels) const
{
  // This is not the best way to do this!

  // We should not have to fill this with values, because each one should be
  // overwritten.
  labels.set_size(observations.n_cols);
  for (size_t i = 0; i < observations.n_cols; ++i)
  {
    // Find maximum probability component.
    double probability = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < gaussians; ++j)
    {
      // We have to use LogProbability() otherwise Probability() would overflow
      // easily.
      double newProb = LogProbability(observations.unsafe_col(i), j);
      if (newProb >= probability)
      {
        probability = newProb;
        labels[i] = j;
      }
    }
  }
}

/**
 * Get the log-likelihood of this data's fit to the model.
 */
double GMM::LogLikelihood(
    const arma::mat& data,
    const std::vector<distribution::GaussianDistribution>& distsL,
    const arma::vec& weightsL) const
{
  double loglikelihood = 0;
  arma::vec logPhis;
  arma::mat logLikelihoods(gaussians, data.n_cols);

  // It has to be LogProbability() otherwise Probability() would overflow easily
  for (size_t i = 0; i < gaussians; ++i)
  {
    distsL[i].LogProbability(data, logPhis);
    logLikelihoods.row(i) = log(weightsL(i)) + trans(logPhis);
  }

  // Now sum over every point.
  for (size_t j = 0; j < data.n_cols; ++j)
    loglikelihood += mlpack::math::AccuLog(logLikelihoods.col(j));
  return loglikelihood;
}

} // namespace gmm
} // namespace mlpack
