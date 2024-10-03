/**
 * @author Kim SangYeon
 * @file methods/gmm/diagonal_gmm_impl.hpp
 *
 * Implementation of template-based DiagonalGMM methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_GMM_DIAGONAL_GMM_IMPL_HPP
#define MLPACK_METHODS_GMM_DIAGONAL_GMM_IMPL_HPP

// In case it hasn't already been included.
#include "diagonal_gmm.hpp"

namespace mlpack {

//! Fit the DiagonalGMM to the given observations.
template<typename FittingType>
double DiagonalGMM::Train(const arma::mat& observations,
                          const size_t trials,
                          const bool useExistingModel,
                          FittingType fitter)
{
  double bestLikelihood; // This will be reported later.

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the
    // DiagonalGMM was initialized with no parameters (0 gaussians,
    // dimensionality of 0).
    fitter.Estimate(observations, dists, weights, useExistingModel);
    bestLikelihood = LogLikelihood(observations, dists, weights);
  }
  else
  {
    if (trials == 0)
      return -DBL_MAX; // It's what they asked for...

    // If each trial must start from the same initial location,
    // we must save it.
    std::vector<DiagonalGaussianDistribution<>> distsOrig;
    arma::vec weightsOrig;
    if (useExistingModel)
    {
      distsOrig = dists;
      weightsOrig = weights;
    }

    // We need to keep temporary copies.  We'll do the first training into the
    // actual model position, so that if it's the best we don't need to
    // copy it.
    fitter.Estimate(observations, dists, weights, useExistingModel);
    bestLikelihood = LogLikelihood(observations, dists, weights);

    Log::Info << "DiagonalGMM::Train(): Log-likelihood of trial 0 is "
        << bestLikelihood << "." << std::endl;

    // Now the temporary model.
    std::vector<DiagonalGaussianDistribution<>> distsTrial(
        gaussians, DiagonalGaussianDistribution<>(dimensionality));
    arma::vec weightsTrial(gaussians);

    for (size_t trial = 1; trial < trials; ++trial)
    {
      if (useExistingModel)
      {
        distsTrial = distsOrig;
        weightsTrial = weightsOrig;
      }

      fitter.Estimate(observations, distsTrial, weightsTrial,
          useExistingModel);

      // Check to see if the log-likelihood of this one is better.
      double newLikelihood = LogLikelihood(observations, distsTrial,
          weightsTrial);

      Log::Info << "DiagonalGMM::Train(): Log-likelihood of trial " << trial
          << " is " << newLikelihood << "." << std::endl;

      if (newLikelihood > bestLikelihood)
      {
        // Save new likelihood and copy new model.
        bestLikelihood = newLikelihood;

        dists = distsTrial;
        weights = weightsTrial;
      }
    }
  }

  // Report final log-likelihood and return it.
  Log::Info << "DiagonalGMM::Train(): log-likelihood of trained GMM is "
      << bestLikelihood << "." << std::endl;
  return bestLikelihood;
}

/**
 * Create a DiagonalGMM with the given number of Gaussians, each of which have
 * the specified dimensionality.  The means and covariances will be set to 0.
 *
 * @param gaussians Number of Gaussians in this GMM.
 * @param dimensionality Dimensionality of each Gaussian.
 */
inline DiagonalGMM::DiagonalGMM(
    const size_t gaussians,
    const size_t dimensionality) :
    gaussians(gaussians),
    dimensionality(dimensionality),
    dists(gaussians,
    DiagonalGaussianDistribution<>(dimensionality)),
    weights(gaussians)
{
  // Set equal weights. Technically this model is still valid, but only barely.
  weights.fill(1.0 / gaussians);
}

// Copy constructor for when the other GMM uses the same fitting type.
inline DiagonalGMM::DiagonalGMM(const DiagonalGMM& other) :
    gaussians(other.Gaussians()),
    dimensionality(other.dimensionality),
    dists(other.dists),
    weights(other.weights) { /* Nothing to do. */ }

inline DiagonalGMM& DiagonalGMM::operator=(const DiagonalGMM& other)
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
inline double DiagonalGMM::LogProbability(const arma::vec& observation) const
{
  // Sum the probability for each Gaussian in our mixture (and we have to
  // multiply by the prior for each Gaussian too).
  double sum = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < gaussians; ++i)
  {
    sum = LogAdd(sum, std::log(weights[i]) +
        dists[i].LogProbability(observation));
  }
  return sum;
}

/**
 * Return the log probability of the given observation GMM matrix.
 *
 * @param observation Observation matrix to compute log-probabilty.
 * @param logProbs Stores the value of log-probability for input.
 */
inline void DiagonalGMM::LogProbability(const arma::mat& observation,
                                        arma::vec& logProbs) const
{
  // Sum the probability for each Gaussian in our mixture (and we have to
  // multiply by the prior for each Gaussian too).
  logProbs.set_size(observation.n_cols);

  // Store log-probability value in a matrix.
  arma::mat logProb(observation.n_cols, gaussians);

  // Assign value to the matrix.
  for (size_t i = 0; i < gaussians; i++)
  {
    arma::vec temp(logProb.colptr(i), observation.n_cols, false, true);
    dists[i].LogProbability(observation, temp);
  }

  // Save log(weights) as a vector.
  arma::vec logWeights = log(weights);

  // Compute log-probability.
  logProb += repmat(logWeights.t(), logProb.n_rows, 1);
  LogSumExp(logProb, logProbs);
}

/**
 * Return the probability of the given observation being from this GMM.
 */
inline double DiagonalGMM::Probability(const arma::vec& observation) const
{
  return std::exp(LogProbability(observation));
}

/**
 * Return the probability of the given observation GMM matrix.
 *
 * @param observation Observation matrix to compute probabilty.
 * @param probs Stores the value of probability for observation.
 */
inline void DiagonalGMM::Probability(const arma::mat& observation,
                                     arma::vec& probs) const
{
  LogProbability(observation, probs);
  probs = exp(probs);
}


/**
 * Return the log probability of the given observation being from the given
 * component in the mixture.
 */
inline double DiagonalGMM::LogProbability(const arma::vec& observation,
                                          const size_t component) const
{
  // We are only considering one Gaussian component -- so we only need to call
  // Probability() once.  We do consider the prior probability!
  return std::log(weights[component]) +
         dists[component].LogProbability(observation);
}

/**
 * Return the probability of the given observation being from the given
 * component in the mixture.
 */
inline double DiagonalGMM::Probability(const arma::vec& observation,
                                       const size_t component) const
{
  return std::exp(LogProbability(observation, component));
}

/**
 * Return a randomly generated observation according to the probability
 * distribution defined by this object.
 */
inline arma::vec DiagonalGMM::Random() const
{
  // Determine which Gaussian it will be coming from.
  double gaussRand = mlpack::Random();
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

  return sqrt(dists[gaussian].Covariance()) %
      randn<arma::vec>(dimensionality) + dists[gaussian].Mean();
}

/**
 * Classify the given observations as being from an individual component in
 * this GMM.
 */
inline void DiagonalGMM::Classify(const arma::mat& observations,
                                  arma::Row<size_t>& labels) const
{
  // This is not the best way to do this!

  // We should not have to fill this with values, because each one should be
  // overwritten.
  labels.set_size(observations.n_cols);
  for (size_t i = 0; i < observations.n_cols; ++i)
  {
    // Find maximum probability component.
    double probability = 0;
    for (size_t j = 0; j < gaussians; ++j)
    {
      double newProb = Probability(observations.unsafe_col(i), j);
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
inline double DiagonalGMM::LogLikelihood(
    const arma::mat& observations,
    const std::vector<DiagonalGaussianDistribution<>>& dists,
    const arma::vec& weights) const
{
  double logLikelihood = 0;
  arma::vec phis;
  arma::mat likelihoods(gaussians, observations.n_cols);

  for (size_t i = 0; i < gaussians; ++i)
  {
    dists[i].Probability(observations, phis);
    likelihoods.row(i) = weights(i) * trans(phis);
  }

  // Now sum over every point.
  for (size_t j = 0; j < observations.n_cols; ++j)
  {
    if (accu(likelihoods.col(j)) == 0)
      Log::Info << "Likelihood of point " << j << " is 0!  It is probably an "
          << "outlier." << std::endl;
    logLikelihood += std::log(accu(likelihoods.col(j)));
  }

  return logLikelihood;
}

/**
 * Fit the DiagonalGMM to the given observations, each of which has a certain
 * probability of being from this distribution.
 */
template<typename FittingType>
double DiagonalGMM::Train(const arma::mat& observations,
                          const arma::vec& probabilities,
                          const size_t trials,
                          const bool useExistingModel,
                          FittingType fitter)
{
  double bestLikelihood; // This will be reported later.

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the
    // DiagonalGMM was initialized with no parameters (0 gaussians,
    // dimensionality of 0).
    fitter.Estimate(observations, probabilities, dists, weights,
        useExistingModel);

    bestLikelihood = LogLikelihood(observations, dists, weights);
  }
  else
  {
    if (trials == 0)
      return -DBL_MAX; // It's what they asked for...

    // If each trial must start from the same initial location, we must save it.
    std::vector<DiagonalGaussianDistribution<>> distsOrig;
    arma::vec weightsOrig;
    if (useExistingModel)
    {
      distsOrig = dists;
      weightsOrig = weights;
    }

    // We need to keep temporary copies.  We'll do the first training into the
    // actual model position, so that if it's the best we don't need to copy it.
    fitter.Estimate(observations, probabilities, dists, weights,
        useExistingModel);

    bestLikelihood = LogLikelihood(observations, dists, weights);

    Log::Debug << "DiagonalGMM::Train(): Log-likelihood of trial 0 is "
        << bestLikelihood << "." << std::endl;

    // Now the temporary model.
    std::vector<DiagonalGaussianDistribution<>> distsTrial(gaussians,
        DiagonalGaussianDistribution<>(dimensionality));
    arma::vec weightsTrial(gaussians);

    for (size_t trial = 1; trial < trials; ++trial)
    {
      if (useExistingModel)
      {
        distsTrial = distsOrig;
        weightsTrial = weightsOrig;
      }

      fitter.Estimate(observations, probabilities, distsTrial, weightsTrial,
          useExistingModel);

      // Check to see if the log-likelihood of this one is better.
      double newLikelihood = LogLikelihood(observations, distsTrial,
          weightsTrial);

      Log::Debug << "DiagonalGMM::Train(): Log-likelihood of trial " << trial
          << " is " << newLikelihood << "." << std::endl;

      if (newLikelihood > bestLikelihood)
      {
        // Save new likelihood and copy new model.
        bestLikelihood = newLikelihood;

        dists = distsTrial;
        weights = weightsTrial;
      }
    }
  }

  // Report final log-likelihood and return it.
  Log::Info << "DiagonalGMM::Train(): log-likelihood of trained GMM is "
      << bestLikelihood << "." << std::endl;
  return bestLikelihood;
}

//! Serialize the object.
template<typename Archive>
void DiagonalGMM::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(gaussians));
  ar(CEREAL_NVP(dimensionality));
  ar(CEREAL_NVP(dists));
  ar(CEREAL_NVP(weights));
}

} // namespace mlpack

#endif // MLPACK_METHODS_GMM_DIAGONAL_GMM_IMPL_HPP
