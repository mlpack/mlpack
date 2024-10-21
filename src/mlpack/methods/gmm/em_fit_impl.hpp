/**
 * @file methods/gmm/em_fit_impl.hpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Implementation of EM algorithm for fitting GMMs.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GMM_EM_FIT_IMPL_HPP
#define MLPACK_METHODS_GMM_EM_FIT_IMPL_HPP

// In case it hasn't been included yet.
#include "em_fit.hpp"
#include "diagonal_constraint.hpp"
#include <mlpack/core/math/log_add.hpp>

namespace mlpack {

//! Constructor.
template<typename InitialClusteringType,
         typename CovarianceConstraintPolicy,
         typename Distribution>
EMFit<InitialClusteringType, CovarianceConstraintPolicy, Distribution>::EMFit(
    const size_t maxIterations,
    const double tolerance,
    InitialClusteringType clusterer,
    CovarianceConstraintPolicy constraint) :
    maxIterations(maxIterations),
    tolerance(tolerance),
    clusterer(clusterer),
    constraint(constraint)
{ /* Nothing to do. */ }

template<typename InitialClusteringType,
         typename CovarianceConstraintPolicy,
         typename Distribution>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy, Distribution>::
Estimate(const arma::mat& observations,
         std::vector<Distribution>& dists,
         arma::vec& weights,
         const bool useInitialModel)
{
  if (std::is_same_v<Distribution, DiagonalGaussianDistribution<>>)
  {
    #ifdef _WIN32
      Log::Warn << "Cannot use arma::gmm_diag on Visual Studio due to OpenMP"
          << " compilation issues! Using slower EMFit::Estimate() instead..."
          << std::endl;
    #else
      ArmadilloGMMWrapper(observations, dists, weights, useInitialModel);
      return;
    #endif
  }
  else if (std::is_same_v<CovarianceConstraintPolicy, DiagonalConstraint>
      && std::is_same_v<Distribution, GaussianDistribution<>>)
  {
    // EMFit::Estimate() using DiagonalConstraint with GaussianDistribution
    // makes use of slower implementation.
    Log::Warn << "EMFit::Estimate() using DiagonalConstraint with "
        << "GaussianDistribution makes use of slower implementation, so "
        << "DiagonalGMM is recommended for faster training." << std::endl;
  }

  // Only perform initial clustering if the user wanted it.
  if (!useInitialModel)
    InitialClustering(observations, dists, weights);

  double l = LogLikelihood(observations, dists, weights);

  Log::Debug << "EMFit::Estimate(): initial clustering log-likelihood: "
      << l << std::endl;

  double lOld = -DBL_MAX;
  arma::mat condLogProb(observations.n_cols, dists.size());

  // Iterate to update the model until no more improvement is found.
  size_t iteration = 1;
  while (std::abs(l - lOld) > tolerance && iteration != maxIterations)
  {
    Log::Info << "EMFit::Estimate(): iteration " << iteration << ", "
        << "log-likelihood " << l << "." << std::endl;

    // Calculate the conditional probabilities of choosing a particular
    // Gaussian given the observations and the present theta value.
    for (size_t i = 0; i < dists.size(); ++i)
    {
      // Store conditional log probabilities into condLogProb vector for each
      // Gaussian.  First we make an alias of the condLogProb vector.
      arma::vec condLogProbAlias = condLogProb.unsafe_col(i);
      dists[i].LogProbability(observations, condLogProbAlias);
      condLogProbAlias += std::log(weights[i]);
    }

    // Normalize row-wise.
    for (size_t i = 0; i < condLogProb.n_rows; ++i)
    {
      // Avoid dividing by zero; if the probability for everything is 0, we
      // don't want to make it NaN.
      const double probSum = AccuLog(condLogProb.row(i));
      if (probSum != -std::numeric_limits<double>::infinity())
        condLogProb.row(i) -= probSum;
    }

    // Store the sum of the probability of each state over all the observations.
    arma::vec probRowSums(dists.size());
    for (size_t i = 0; i < dists.size(); ++i)
    {
      probRowSums(i) = AccuLog(condLogProb.col(i));
    }

    // Calculate the new value of the means using the updated conditional
    // probabilities.
    for (size_t i = 0; i < dists.size(); ++i)
    {
      // Don't update if there's no probability of the Gaussian having points.
      if (probRowSums[i] != -std::numeric_limits<double>::infinity())
        dists[i].Mean() = observations * exp(condLogProb.col(i) -
                                                   probRowSums[i]);
      else
        continue;

      // Calculate the new value of the covariances using the updated
      // conditional probabilities and the updated means.
      arma::mat tmp = observations.each_col() - dists[i].Mean();

      // If the distribution is DiagonalGaussianDistribution, calculate the
      // covariance only with diagonal components.
      if (std::is_same_v<Distribution, DiagonalGaussianDistribution<>>)
      {
        arma::vec covariance = sum((tmp % tmp) %
            (ones<arma::vec>(observations.n_rows) *
            trans(exp(condLogProb.col(i) - probRowSums[i]))), 1);

        // Apply covariance constraint.
        constraint.ApplyConstraint(covariance);
        dists[i].Covariance(std::move(covariance));
      }
      else
      {
        arma::mat tmpB = tmp.each_row() % trans(exp(condLogProb.col(i) -
                                                          probRowSums[i]));
        arma::mat covariance = tmp * trans(tmpB);

        // Apply covariance constraint.
        constraint.ApplyConstraint(covariance);
        dists[i].Covariance(std::move(covariance));
      }
    }

    // Calculate the new values for omega using the updated conditional
    // probabilities.
    weights = exp(probRowSums - std::log(observations.n_cols));

    // Update values of l; calculate new log-likelihood.
    lOld = l;
    l = LogLikelihood(observations, dists, weights);

    iteration++;
  }
}

template<typename InitialClusteringType,
         typename CovarianceConstraintPolicy,
         typename Distribution>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy, Distribution>::
Estimate(const arma::mat& observations,
         const arma::vec& probabilities,
         std::vector<Distribution>& dists,
         arma::vec& weights,
         const bool useInitialModel)
{
  if (!useInitialModel)
    InitialClustering(observations, dists, weights);

  double l = LogLikelihood(observations, dists, weights);

  Log::Debug << "EMFit::Estimate(): initial clustering log-likelihood: "
      << l << std::endl;

  double lOld = -DBL_MAX;
  arma::mat condLogProb(observations.n_cols, dists.size());

  // Iterate to update the model until no more improvement is found.
  size_t iteration = 1;
  while (std::abs(l - lOld) > tolerance && iteration != maxIterations)
  {
    // Calculate the conditional probabilities of choosing a particular
    // Gaussian given the observations and the present theta value.
    for (size_t i = 0; i < dists.size(); ++i)
    {
      // Store conditional log probabilities into condLogProb vector for each
      // Gaussian.  First we make an alias of the condLogProb vector.
      arma::vec condLogProbAlias = condLogProb.unsafe_col(i);
      dists[i].LogProbability(observations, condLogProbAlias);
      condLogProbAlias += std::log(weights[i]);
    }

    // Normalize row-wise.
    for (size_t i = 0; i < condLogProb.n_rows; ++i)
    {
      // Avoid dividing by zero; if the probability for everything is 0, we
      // don't want to make it NaN.
      const double probSum = AccuLog(condLogProb.row(i));
      if (probSum != -std::numeric_limits<double>::infinity())
        condLogProb.row(i) -= probSum;
    }

    // This will store the sum of probabilities of each state over all the
    // observations.
    arma::vec probRowSums(dists.size());

    // Calculate the new value of the means using the updated conditional
    // probabilities.
    arma::vec logProbabilities = log(probabilities);
    for (size_t i = 0; i < dists.size(); ++i)
    {
      // Calculate the sum of probabilities of points, which is the
      // conditional probability of each point being from Gaussian i
      // multiplied by the probability of the point being from this mixture
      // model.
      arma::vec tmpProb = condLogProb.col(i) + logProbabilities;
      probRowSums[i] = AccuLog(tmpProb);

      // Don't update if there's no probability of the Gaussian having points.
      if (probRowSums[i] != -std::numeric_limits<double>::infinity())
      {
        dists[i].Mean() = observations *
              exp(condLogProb.col(i) + logProbabilities - probRowSums[i]);
      }
      else
        continue;

      // Calculate the new value of the covariances using the updated
      // conditional probabilities and the updated means.
      arma::mat tmp = observations.each_col() - dists[i].Mean();

      // If the distribution is DiagonalGaussianDistribution, calculate the
      // covariance only with diagonal components.
      if (std::is_same_v<Distribution, DiagonalGaussianDistribution<>>)
      {
        arma::vec cov = sum((tmp % tmp) %
            (ones<arma::vec>(observations.n_rows) *
            trans(exp(condLogProb.col(i) +
                            logProbabilities - probRowSums[i]))), 1);

        // Apply covariance constraint.
        constraint.ApplyConstraint(cov);
        dists[i].Covariance(std::move(cov));
      }
      else
      {
        arma::mat tmpB = tmp.each_row() % trans(exp(condLogProb.col(i) +
            logProbabilities - probRowSums[i]));
        arma::mat cov = (tmp * trans(tmpB));

        // Apply covariance constraint.
        constraint.ApplyConstraint(cov);
        dists[i].Covariance(std::move(cov));
      }
    }

    // Calculate the new values for omega using the updated conditional
    // probabilities.
    weights = exp(probRowSums - AccuLog(logProbabilities));

    // Update values of l; calculate new log-likelihood.
    lOld = l;
    l = LogLikelihood(observations, dists, weights);

    iteration++;
  }
}

template<typename InitialClusteringType,
         typename CovarianceConstraintPolicy,
         typename Distribution>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy, Distribution>::
InitialClustering(const arma::mat& observations,
                  std::vector<Distribution>& dists,
                  arma::vec& weights)
{
  // Assignments from clustering.
  arma::Row<size_t> assignments;

  // Run clustering algorithm.
  clusterer.Cluster(observations, dists.size(), assignments);

  // Check if the type of Distribution is DiagonalGaussianDistribution.  If so,
  // we can get faster performance by using diagonal elements when calculating
  // the covariance.
  const bool isDiagGaussDist = std::is_same_v<Distribution,
      DiagonalGaussianDistribution<>>;

  std::vector<arma::vec> means(dists.size());

  // Conditional covariance instantiation.
  std::vector<std::conditional_t<isDiagGaussDist, arma::vec, arma::mat>>
      covs(dists.size());

  // Now calculate the means, covariances, and weights.
  weights.zeros();
  for (size_t i = 0; i < dists.size(); ++i)
  {
    means[i].zeros(dists[i].Mean().n_elem);
    if (isDiagGaussDist)
    {
      covs[i].zeros(dists[i].Covariance().n_elem);
    }
    else
    {
      covs[i].zeros(dists[i].Covariance().n_rows,
                    dists[i].Covariance().n_cols);
    }
  }

  // From the assignments, generate our means, covariances, and weights.
  for (size_t i = 0; i < observations.n_cols; ++i)
  {
    const size_t cluster = assignments[i];

    // Add this to the relevant mean.
    means[cluster] += observations.col(i);

    // Add this to the relevant covariance.
    if (isDiagGaussDist)
      covs[cluster] += observations.col(i) % observations.col(i);
    else
      covs[cluster] += observations.col(i) * trans(observations.col(i));

    // Now add one to the weights (we will normalize).
    weights[cluster]++;
  }

  // Now normalize the mean and covariance.
  for (size_t i = 0; i < dists.size(); ++i)
  {
    means[i] /= (weights[i] > 1) ? weights[i] : 1;
  }

  for (size_t i = 0; i < observations.n_cols; ++i)
  {
    const size_t cluster = assignments[i];
    const arma::vec normObs = observations.col(i) - means[cluster];
    if (isDiagGaussDist)
      covs[cluster] += normObs % normObs;
    else
      covs[cluster] += normObs * normObs.t();
  }

  for (size_t i = 0; i < dists.size(); ++i)
  {
    covs[i] /= (weights[i] > 1) ? weights[i] : 1;

    // Apply constraints to covariance matrix.
    if (isDiagGaussDist)
      covs[i] = arma::clamp(covs[i], 1e-10, DBL_MAX);
    else
      constraint.ApplyConstraint(covs[i]);

    std::swap(dists[i].Mean(), means[i]);
    dists[i].Covariance(std::move(covs[i]));
  }

  // Finally, normalize weights.
  weights /= accu(weights);
}

template<typename InitialClusteringType,
         typename CovarianceConstraintPolicy,
         typename Distribution>
double EMFit<InitialClusteringType, CovarianceConstraintPolicy, Distribution>::
LogLikelihood(const arma::mat& observations,
              const std::vector<Distribution>& dists,
              const arma::vec& weights) const
{
  double logLikelihood = 0;

  arma::vec logPhis;
  arma::mat logLikelihoods(dists.size(), observations.n_cols);

  // It has to be LogProbability() otherwise Probability() would overflow easily
  for (size_t i = 0; i < dists.size(); ++i)
  {
    dists[i].LogProbability(observations, logPhis);
    logLikelihoods.row(i) = std::log(weights(i)) + trans(logPhis);
  }
  // Now sum over every point.
  for (size_t j = 0; j < observations.n_cols; ++j)
  {
    if (AccuLog(logLikelihoods.col(j)) ==
        -std::numeric_limits<double>::infinity())
    {
      Log::Info << "Likelihood of point " << j << " is 0!  It is probably an "
          << "outlier." << std::endl;
    }
    logLikelihood += AccuLog(logLikelihoods.col(j));
  }

  return logLikelihood;
}

template<typename InitialClusteringType,
         typename CovarianceConstraintPolicy,
         typename Distribution>
template<typename Archive>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy, Distribution>::
serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(maxIterations));
  ar(CEREAL_NVP(tolerance));
  ar(CEREAL_NVP(clusterer));
  ar(CEREAL_NVP(constraint));
}

template<typename InitialClusteringType,
         typename CovarianceConstraintPolicy,
         typename Distribution>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy, Distribution>::
ArmadilloGMMWrapper(const arma::mat& observations,
                    std::vector<Distribution>& dists,
                    arma::vec& weights,
                    const bool useInitialModel)
{
  arma::gmm_diag g;

  // Warn the user that tolerance isn't used for convergence here if they've
  // specified a non-default value.
  if (tolerance != EMFit().Tolerance())
    Log::Warn << "GMM::Train(): tolerance ignored when training GMMs with "
        << "DiagonalConstraint." << std::endl;

  // If the initial clustering is the default k-means, we'll just use
  // Armadillo's implementation.  If mlpack ever changes k-means defaults to use
  // something that is reliably quicker than the Lloyd iteration k-means update,
  // then this code maybe should be revisited.
  if (!std::is_same_v<InitialClusteringType, KMeans<>> || useInitialModel)
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

      // DiagonalGaussianDistribution has diagonal covariance as an arma::vec.
      covs.col(i) = dists[i].Covariance();
    }

    g.reset(observations.n_rows, dists.size());
    g.set_params(std::move(means), std::move(covs), weights.t());

    g.learn(observations, dists.size(), arma::eucl_dist, arma::keep_existing, 0,
        maxIterations, 1e-10, false /* no printing */);
  }
  else
  {
    // Use Armadillo for the initial clustering.  We'll try and match mlpack
    // defaults.
    g.learn(observations, dists.size(), arma::eucl_dist, arma::static_subset,
        1000, maxIterations, 1e-10, false /* no printing */);
  }

  // Extract means, covariances, and weights.
  weights = g.hefts.t();
  for (size_t i = 0; i < dists.size(); ++i)
  {
    dists[i].Mean() = g.means.col(i);

    // Apply covariance constraint.
    arma::vec covsAlias = g.dcovs.unsafe_col(i);
    constraint.ApplyConstraint(covsAlias);

    // DiagonalGaussianDistribution has diagonal covariance as an arma::vec.
    dists[i].Covariance(g.dcovs.col(i));
  }
}

} // namespace mlpack

#endif
