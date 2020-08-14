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
namespace gmm {

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
    std::vector<distribution::DiagonalGaussianDistribution> distsOrig;
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
    std::vector<distribution::DiagonalGaussianDistribution> distsTrial(
        gaussians, distribution::DiagonalGaussianDistribution(dimensionality));
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
    std::vector<distribution::DiagonalGaussianDistribution> distsOrig;
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
    std::vector<distribution::DiagonalGaussianDistribution> distsTrial(
        gaussians, distribution::DiagonalGaussianDistribution(dimensionality));
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
void DiagonalGMM::serialize(Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(gaussians);
  ar & CEREAL_NVP(dimensionality);
  ar & CEREAL_NVP(dists);
  ar & CEREAL_NVP(weights);
}

} // namespace gmm
} // namespace mlpack

#endif // MLPACK_METHODS_GMM_DIAGONAL_GMM_IMPL_HPP
