/**
 * @author Kim SangYeon
 * @file diagonal_gmm_impl.hpp
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
  const size_t maxIterations = fitter.MaxIterations();
  const double tolerance = fitter.Tolerance();

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the
    // DiagonalGMM was initialized with no parameters (0 gaussians,
    // dimensionality of 0).
    Estimate(observations, dists, weights, useExistingModel, maxIterations,
        tolerance, fitter.Clusterer());
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
    Estimate(observations, dists, weights, useExistingModel, maxIterations,
        tolerance, fitter.Clusterer());
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

      Estimate(observations, distsTrial, weightsTrial, useExistingModel,
          maxIterations, tolerance, fitter.Clusterer());

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
  const size_t maxIterations = fitter.MaxIterations();
  const double tolerance = fitter.Tolerance();

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the
    // DiagonalGMM was initialized with no parameters (0 gaussians,
    // dimensionality of 0).
    Estimate(observations, probabilities, dists, weights, useExistingModel,
        maxIterations, tolerance, fitter.Clusterer());

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
    Estimate(observations, probabilities, dists, weights, useExistingModel,
        maxIterations, tolerance, fitter.Clusterer());

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

      Estimate(observations, probabilities, distsTrial, weightsTrial,
          useExistingModel, maxIterations, tolerance, fitter.Clusterer());

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

template<typename InitialClusteringType>
void DiagonalGMM::Estimate(
    const arma::mat& observations,
    std::vector<distribution::DiagonalGaussianDistribution>& dists,
    arma::vec& weights,
    const bool useInitialModel,
    const size_t maxIterations,
    const double tolerance,
    InitialClusteringType clusterer)
{
  // If it is not on Windows, we use the Armadillo's gmm_diag class by calling
  // ArmadilloGMMWrapper().
  #ifndef _WIN32
  ArmadilloGMMWrapper(observations, dists, weights, useInitialModel,
      maxIterations, tolerance, clusterer);

  #else
  if (!useInitialModel)
    InitialClustering(observations, dists, weights, clusterer);

  // Set the initial log likelihood.
  double l = LogLikelihood(observations, dists, weights);

  Log::Debug << "DiagonalGMM::Estimate(): initial clustering log likelihood: "
      << l << std::endl;

  // Initialize the old log likelihood for comparison later.
  double lOld = -DBL_MAX;

  // Create the conditional probability matrix.
  arma::mat condProb(observations.n_cols, dists.size());

  // Iterate to update the model until no more improvement is found.
  size_t iteration = 1;
  while (std::abs(l - lOld) > tolerance && iteration != maxIterations)
  {
    // E step: Calculate the conditional probabilities of the given
    // observations choosing a particular gaussian using current parameters.
    for (size_t k = 0; k < dists.size(); k++)
    {
      arma::vec condProbAlias = condProb.unsafe_col(k);
      dists[k].Probability(observations, condProbAlias);
      condProbAlias *= weights[k];
    }

    // Normalize row-wise.
    for (size_t j = 0; j < condProb.n_rows; j++)
    {
      // Avoid dividing by zero.
      const double probSum = accu(condProb.row(j));
      if (probSum != 0.0)
        condProb.row(j) /= probSum;
    }

    // Store the sum of responsibilities over all the observations.
    arma::vec N = arma::trans(arma::sum(condProb));

    // M step: Update the paramters using the current responsibilities.
    for (size_t k = 0; k < dists.size(); k++)
    {
      // Update the mean and covariance using the responsibilities.
      // If N[k] is zero, we don't update them.
      if (N[k] == 0)
        continue;

      // Update the mean of distribution k.
      dists[k].Mean() = (observations * condProb.col(k)) / N[k];

      // Update the diagonal covariance of distribution k.
      // We only need the diagonal elements in the covariances.
      arma::mat diffs = observations - (dists[k].Mean() *
          arma::ones<arma::rowvec>(observations.n_cols));

      arma::vec covs = arma::sum((diffs % diffs) %
          (arma::ones<arma::vec>(observations.n_rows) *
          trans(condProb.col(k))), 1) / N[k];

      covs = arma::clamp(covs, 1e-10, DBL_MAX);
      dists[k].Covariance(std::move(covs));
    }

    // Update the mixing coefficients.
    weights = N / observations.n_cols;

    // Update log likelihood and Keep the old likelihood for comparison.
    lOld = l;
    l = LogLikelihood(observations, dists, weights);

    iteration++;
  }
  #endif
}

template<typename InitialClusteringType>
void DiagonalGMM::Estimate(const arma::mat& observations,
    const arma::vec& probabilities,
    std::vector<distribution::DiagonalGaussianDistribution>& dists,
    arma::vec& weights,
    const bool useInitialModel,
    const size_t maxIterations,
    const double tolerance,
    InitialClusteringType clusterer)
{
  if (!useInitialModel)
    InitialClustering(observations, dists, weights, clusterer);

  // Set the initial log likelihood.
  double l = LogLikelihood(observations, dists, weights);

  Log::Debug << "DiagonalGMM::Estimate(): initial clustering log likelihood: "
      << l << std::endl;

  // Initialze the old log likelihood for comparison later.
  double lOld = -DBL_MAX;

  // Create the conditional probability matrix.
  arma::mat condProb(observations.n_cols, dists.size());

  // Iterate to update the model until no more improvement is found.
  size_t iteration = 1;
  while (std::abs(l - lOld) > tolerance && iteration != maxIterations)
  {
    // E step: Calculate the conditional probabilities of the given
    // observations choosing a particular gaussian using current parameters.
    for (size_t k = 0; k < dists.size(); k++)
    {
      arma::vec condProbAlias = condProb.unsafe_col(k);
      dists[k].Probability(observations, condProbAlias);
      condProbAlias *= weights[k];
    }

    // Normalize row-wise.
    for (size_t j = 0; j < condProb.n_rows; j++)
    {
      // Avoid dividing by zero.
      const double probSum = accu(condProb.row(j));
      if (probSum != 0.0)
        condProb.row(j) /= probSum;
    }

    // Store the sum of responsibilities over all the observations.
    arma::vec N(dists.size());

    // M step: Update the paramters using the current responsibilities.
    for (size_t k = 0; k < dists.size(); k++)
    {
      // Calculate the sum of conditional probabilities of each point, being
      // from Gaussian i, multiplied by the probability of the point.
      N[k] = arma::accu(condProb.col(k) % probabilities);

      // Update the mean and covariance using the responsibilities.
      // If N[k] is zero, we don't update them.
      if (N[k] == 0)
        continue;

      // Update the mean of distribution k.
      dists[k].Mean() = (observations * (condProb.col(k) % probabilities)) /
          N[k];

      // Update the diagonal covariance of distribution k.
      // We only need the diagonal elements in the covariances.
      arma::mat diffs = observations - (dists[k].Mean() *
          arma::ones<arma::rowvec>(observations.n_cols));

      arma::vec covs = arma::sum((diffs % diffs) %
          (arma::ones<arma::vec>(observations.n_rows) *
          trans(condProb.col(k) % probabilities)), 1) / N[k];

      covs = arma::clamp(covs, 1e-10, DBL_MAX);
      dists[k].Covariance(std::move(covs));
    }

    // Update the mixing coefficients.
    weights = N / arma::accu(probabilities);

    // Update log likelihood and Keep the old likelihood for comparison.
    lOld = l;
    l = LogLikelihood(observations, dists, weights);

    iteration++;
  }
}

//! Cluster initially.
template<typename InitialClusteringType>
void DiagonalGMM::InitialClustering(
    const arma::mat& observations,
    std::vector<distribution::DiagonalGaussianDistribution>& dists,
    arma::vec& weights,
    InitialClusteringType clusterer)
{
  // Assignments from clustering.
  arma::Row<size_t> assignments;

  // Run clustering algorithm.
  clusterer.Cluster(observations, dists.size(), assignments);

  std::vector<arma::vec> means(dists.size());
  std::vector<arma::vec> covariances(dists.size());

  // Now calculate the means, covariances, and weights.
  // Initialize the means, covariances, and weights.
  weights.zeros();
  for (size_t i = 0; i < dists.size(); i++)
  {
    means[i].zeros(dists[i].Mean().n_elem);
    covariances[i].zeros(dists[i].Covariance().n_elem);
  }

  // From the assignments, generate means, covariances, and weights.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    const size_t cluster = assignments[i];

    // Add this to the relevant mean.
    means[cluster] += observations.col(i);

    // Add this to the relevant covariance.
    covariances[cluster] += observations.col(i) % observations.col(i);

    // Add one to the weights to normalize parameters later.
    weights[cluster]++;
  }

  // Normalize the mean.
  for (size_t i = 0; i < dists.size(); i++)
  {
    means[i] /= (weights[i] > 1) ? weights[i] : 1;
  }

  // Calculate the covariances.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    const size_t cluster = assignments[i];
    const arma::vec diffs = observations.col(i) - means[cluster];
    covariances[cluster] += diffs % diffs;
  }

  // Normalize the covariances.
  for (size_t i = 0; i < dists.size(); i++)
  {
    covariances[i] /= (weights[i] > 1) ? weights[i] : 1;

    std::swap(dists[i].Mean(), means[i]);

    covariances[i] = arma::clamp(covariances[i], 1e-10, DBL_MAX);
    dists[i].Covariance(std::move(covariances[i]));
  }

  // Normalize weights.
  weights /= arma::accu(weights);
}

// Armadillo uses uword internally as an OpenMP index type, which crashes
// Visual Studio.
#ifndef _WIN32
template<typename InitialClusteringType>
void DiagonalGMM::ArmadilloGMMWrapper(
    const arma::mat& observations,
    std::vector<distribution::DiagonalGaussianDistribution>& dists,
    arma::vec& weights,
    const bool useInitialModel,
    const size_t maxIterations,
    const double tolerance,
    InitialClusteringType clusterer)
{
  arma::gmm_diag gmm;

  // Warn the user that tolerance isn't used for convergence here if they've
  // specified a non-default value.
  if (tolerance != EMFit<>().Tolerance())
    Log::Warn << "DiagonalGMM::Train(): tolerance ignored when training GMMs."
        << std::endl;

  // If the initial clustering is the default k-means, we'll just use
  // Armadillo's implementation.  If mlpack ever changes k-means defaults to use
  // something that is reliably quicker than the Lloyd iteration k-means update,
  // then this code maybe should be revisited.
  if (!std::is_same<InitialClusteringType, mlpack::kmeans::KMeans<>>::value ||
      useInitialModel)
  {
    // Use clusterer to get initial values.
    if (!useInitialModel)
      InitialClustering(observations, dists, weights, clusterer);

    // Assemble matrix of means.
    arma::mat means(observations.n_rows, dists.size());
    arma::mat covs(observations.n_rows, dists.size());
    for (size_t i = 0; i < dists.size(); ++i)
    {
      means.col(i) = dists[i].Mean();
      covs.col(i) = dists[i].Covariance();
    }

    gmm.reset(observations.n_rows, dists.size());
    gmm.set_params(std::move(means), std::move(covs), weights.t());

    gmm.learn(observations, dists.size(), arma::eucl_dist, arma::keep_existing,
        0, maxIterations, 1e-10, false /* no printing */);
  }
  else
  {
    // Use Armadillo for the initial clustering.  We'll try and match mlpack
    // defaults.
    gmm.learn(observations, dists.size(), arma::eucl_dist, arma::static_subset,
        1000, maxIterations, 1e-10, false /* no printing */);
  }

  // Extract means, covariances, and weights.
  weights = gmm.hefts.t();
  for (size_t i = 0; i < dists.size(); ++i)
  {
    dists[i].Mean() = gmm.means.col(i);
    dists[i].Covariance(gmm.dcovs.col(i));
  }
}
#endif

//! Serialize the object.
template<typename Archive>
void DiagonalGMM::serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(gaussians);
  ar & BOOST_SERIALIZATION_NVP(dimensionality);
  ar & BOOST_SERIALIZATION_NVP(dists);
  ar & BOOST_SERIALIZATION_NVP(weights);
}

} // namespace gmm
} // namespace mlpack

#endif // MLPACK_METHODS_GMM_DIAGONAL_GMM_IMPL_HPP
