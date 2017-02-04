/**
 * @file gmm_impl.hpp
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
#ifndef MLPACK_METHODS_GMM_GMM_IMPL_HPP
#define MLPACK_METHODS_GMM_GMM_IMPL_HPP

// In case it hasn't already been included.
#include "gmm.hpp"

namespace mlpack {
namespace gmm {

/**
 * Fit the GMM to the given observations.
 */
template<typename FittingType>
double GMM::Train(const arma::mat& observations,
                  const size_t trials,
                  const bool useExistingModel,
                  FittingType fitter)
{
  double bestLikelihood; // This will be reported later.

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the GMM was
    // initialized with no parameters (0 gaussians, dimensionality of 0).
    fitter.Estimate(observations, dists, weights, useExistingModel);
    bestLikelihood = LogLikelihood(observations, dists, weights);
  }
  else
  {
    if (trials == 0)
      return -DBL_MAX; // It's what they asked for...

    // If each trial must start from the same initial location, we must save it.
    std::vector<distribution::GaussianDistribution> distsOrig;
    arma::vec weightsOrig;
    if (useExistingModel)
    {
      distsOrig = dists;
      weightsOrig = weights;
    }

    // We need to keep temporary copies.  We'll do the first training into the
    // actual model position, so that if it's the best we don't need to copy it.
    fitter.Estimate(observations, dists, weights, useExistingModel);

    bestLikelihood = LogLikelihood(observations, dists, weights);

    Log::Info << "GMM::Train(): Log-likelihood of trial 0 is " << bestLikelihood        << "." << std::endl;

    // Now the temporary model.
    std::vector<distribution::GaussianDistribution> distsTrial(gaussians,
        distribution::GaussianDistribution(dimensionality));
    arma::vec weightsTrial(gaussians);

    for (size_t trial = 1; trial < trials; ++trial)
    {
      if (useExistingModel)
      {
        distsTrial = distsOrig;
        weightsTrial = weightsOrig;
      }

      fitter.Estimate(observations, distsTrial, weightsTrial, useExistingModel);

      // Check to see if the log-likelihood of this one is better.
      double newLikelihood = LogLikelihood(observations, distsTrial,
          weightsTrial);

      Log::Info << "GMM::Train(): Log-likelihood of trial " << trial << " is "
          << newLikelihood << "." << std::endl;

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
  Log::Info << "GMM::Train(): log-likelihood of trained GMM is "
      << bestLikelihood << "." << std::endl;
  return bestLikelihood;
}

/**
 * Fit the GMM to the given observations, each of which has a certain
 * probability of being from this distribution.
 */
template<typename FittingType>
double GMM::Train(const arma::mat& observations,
                  const arma::vec& probabilities,
                  const size_t trials,
                  const bool useExistingModel,
                  FittingType fitter)
{
  double bestLikelihood; // This will be reported later.

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the GMM was
    // initialized with no parameters (0 gaussians, dimensionality of 0).
    fitter.Estimate(observations, probabilities, dists, weights,
        useExistingModel);
    bestLikelihood = LogLikelihood(observations, dists, weights);
  }
  else
  {
    if (trials == 0)
      return -DBL_MAX; // It's what they asked for...

    // If each trial must start from the same initial location, we must save it.
    std::vector<distribution::GaussianDistribution> distsOrig;
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

    Log::Debug << "GMM::Train(): Log-likelihood of trial 0 is "
        << bestLikelihood << "." << std::endl;

    // Now the temporary model.
    std::vector<distribution::GaussianDistribution> distsTrial(gaussians,
        distribution::GaussianDistribution(dimensionality));
    arma::vec weightsTrial(gaussians);

    for (size_t trial = 1; trial < trials; ++trial)
    {
      if (useExistingModel)
      {
        distsTrial = distsOrig;
        weightsTrial = weightsOrig;
      }

      fitter.Estimate(observations, probabilities, distsTrial, weightsTrial, useExistingModel);

      // Check to see if the log-likelihood of this one is better.
      double newLikelihood = LogLikelihood(observations, distsTrial,
          weightsTrial);

      Log::Debug << "GMM::Train(): Log-likelihood of trial " << trial << " is "
          << newLikelihood << "." << std::endl;

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
  Log::Info << "GMM::Train(): log-likelihood of trained GMM is "
      << bestLikelihood << "." << std::endl;
  return bestLikelihood;
}

/**
 * Serialize the object.
 */
template<typename Archive>
void GMM::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(gaussians, "gaussians");
  ar & CreateNVP(dimensionality, "dimensionality");

  // Load (or save) the gaussians.  Not going to use the default std::vector
  // serialize here because it won't call out correctly to Serialize() for each
  // Gaussian distribution.
  if (Archive::is_loading::value)
    dists.resize(gaussians);

  for (size_t i = 0; i < gaussians; ++i)
  {
    std::ostringstream oss;
    oss << "dist" << i;
    ar & CreateNVP(dists[i], oss.str());
  }

  ar & CreateNVP(weights, "weights");
}

} // namespace gmm
} // namespace mlpack

#endif

