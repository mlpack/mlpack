/**
 * @file diagonal_em_fit_impl.hpp
 * @author Kim SangYeon
 *
 * Implementation of EM algorithm for fitting DiagonalGMMs.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GMM_DIAGONAL_EM_FIT_IMPL_HPP
#define MLPACK_METHODS_GMM_DIAGONAL_EM_FIT_IMPL_HPP

// In case it hasn't been included yet.
#include "diagonal_em_fit.hpp"
#include "no_constraint.hpp"

namespace mlpack {
namespace gmm {

//! Constructor.
template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
DiagonalEMFit<InitialClusteringType, CovarianceConstraintPolicy>::
DiagonalEMFit(const size_t maxIterations,
              const double tolerance,
              InitialClusteringType clusterer,
              CovarianceConstraintPolicy constraint) :
              maxIterations(maxIterations),
              tolerance(tolerance),
              clusterer(clusterer),
              constraint(constraint)
{ /* Nothing to do. */ }

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
void DiagonalEMFit<InitialClusteringType, CovarianceConstraintPolicy>::
Estimate(const arma::mat& observations,
         std::vector<distribution::DiagonalGaussianDistribution>& dists,
         arma::vec& weights,
         const bool useInitialModel)
{
  // If it is not on Windows, we use the Armadillo's gmm_diag class by calling
  // ArmadilloGMMWrapper().
  #ifndef _WIN32
  ArmadilloGMMWrapper(observations, dists, weights, useInitialModel);

  #else
  if (!useInitialModel)
    InitialClustering(observations, dists, weights);

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

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
void DiagonalEMFit<InitialClusteringType, CovarianceConstraintPolicy>::
Estimate(const arma::mat& observations,
         const arma::vec& probabilities,
         std::vector<distribution::DiagonalGaussianDistribution>& dists,
         arma::vec& weights,
         const bool useInitialModel)
{
  if (!useInitialModel)
    InitialClustering(observations, dists, weights);

  // Set the initial log likelihood.
  double l = LogLikelihood(observations, dists, weights);

  Log::Debug << "DiagonalEMFit::Estimate(): initial clustering log likelihood:"
      << " " << l << std::endl;

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

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
void DiagonalEMFit<InitialClusteringType, CovarianceConstraintPolicy>::
InitialClustering(
	  const arma::mat& observations,
    std::vector<distribution::DiagonalGaussianDistribution>& dists,
    arma::vec& weights)
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

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
double DiagonalEMFit<InitialClusteringType, CovarianceConstraintPolicy>::
LogLikelihood(
    const arma::mat& observations,
    const std::vector<distribution::DiagonalGaussianDistribution>& dists,
    const arma::vec& weights) const
{
  double logLikelihood = 0;

  arma::vec phis;
  arma::mat likelihoods(dists.size(), observations.n_cols);

  for (size_t i = 0; i < dists.size(); ++i)
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
    logLikelihood += log(accu(likelihoods.col(j)));
  }

  return logLikelihood;
}

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
template<typename Archive>
void DiagonalEMFit<InitialClusteringType, CovarianceConstraintPolicy>::
serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(maxIterations);
  ar & BOOST_SERIALIZATION_NVP(tolerance);
  ar & BOOST_SERIALIZATION_NVP(clusterer);
  ar & BOOST_SERIALIZATION_NVP(constraint);
}

// Armadillo uses uword internally as an OpenMP index type, which crashes
// Visual Studio.
#ifndef _WIN32
template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
void DiagonalEMFit<InitialClusteringType, CovarianceConstraintPolicy>::
ArmadilloGMMWrapper(
	  const arma::mat& observations,
    std::vector<distribution::DiagonalGaussianDistribution>& dists,
    arma::vec& weights,
    const bool useInitialModel)
{
  arma::gmm_diag gmm;

  // Warn the user that tolerance isn't used for convergence here if they've
  // specified a non-default value.
  if (tolerance != DiagonalEMFit<>().Tolerance())
    Log::Warn << "DiagonalGMM::Train(): tolerance ignored when training GMMs."
        << std::endl;

  // If the initial clustering is the default k-means, we'll just use
  // Armadillo's implementation.  If mlpack ever changes k-means defaults to
	// use something that is reliably quicker than the Lloyd iteration k-means
	// update, then this code maybe should be revisited.
  if (!std::is_same<InitialClusteringType, mlpack::kmeans::KMeans<>>::value ||
      useInitialModel)
  {
    // Use clusterer to get initial values.
    if (!useInitialModel)
      InitialClustering(observations, dists, weights);

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

} // namespace gmm
} // namespace mlpack

#endif
