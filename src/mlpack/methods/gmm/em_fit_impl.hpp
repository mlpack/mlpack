/**
 * @file em_fit_impl.hpp
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

namespace mlpack {
namespace gmm {

//! Constructor.
template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
EMFit<InitialClusteringType, CovarianceConstraintPolicy>::EMFit(
    const size_t maxIterations,
    const double tolerance,
    InitialClusteringType clusterer,
    CovarianceConstraintPolicy constraint) :
    maxIterations(maxIterations),
    tolerance(tolerance),
    clusterer(clusterer),
    constraint(constraint)
{ /* Nothing to do. */ }

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy>::Estimate(
    const arma::mat& observations,
    std::vector<distribution::GaussianDistribution>& dists,
    arma::vec& weights,
    const bool useInitialModel)
{
  // Only perform initial clustering if the user wanted it.
  if (!useInitialModel)
    InitialClustering(observations, dists, weights);

  double l = LogLikelihood(observations, dists, weights);

  Log::Debug << "EMFit::Estimate(): initial clustering log-likelihood: "
      << l << std::endl;

  double lOld = -DBL_MAX;
  arma::mat condProb(observations.n_cols, dists.size());

  // Iterate to update the model until no more improvement is found.
  size_t iteration = 1;
  while (std::abs(l - lOld) > tolerance && iteration != maxIterations)
  {
    Log::Info << "EMFit::Estimate(): iteration " << iteration << ", "
        << "log-likelihood " << l << "." << std::endl;

    // Calculate the conditional probabilities of choosing a particular
    // Gaussian given the observations and the present theta value.
    for (size_t i = 0; i < dists.size(); i++)
    {
      // Store conditional probabilities into condProb vector for each
      // Gaussian.  First we make an alias of the condProb vector.
      arma::vec condProbAlias = condProb.unsafe_col(i);
      dists[i].Probability(observations, condProbAlias);
      condProbAlias *= weights[i];
    }

    // Normalize row-wise.
    for (size_t i = 0; i < condProb.n_rows; i++)
    {
      // Avoid dividing by zero; if the probability for everything is 0, we
      // don't want to make it NaN.
      const double probSum = accu(condProb.row(i));
      if (probSum != 0.0)
        condProb.row(i) /= probSum;
    }

    // Store the sum of the probability of each state over all the observations.
    arma::vec probRowSums = trans(arma::sum(condProb, 0 /* columnwise */));

    // Calculate the new value of the means using the updated conditional
    // probabilities.
    for (size_t i = 0; i < dists.size(); i++)
    {
      // Don't update if there's no probability of the Gaussian having points.
      if (probRowSums[i] != 0)
        dists[i].Mean() = (observations * condProb.col(i)) / probRowSums[i];

      // Calculate the new value of the covariances using the updated
      // conditional probabilities and the updated means.
      arma::mat tmp = observations - (dists[i].Mean() *
          arma::ones<arma::rowvec>(observations.n_cols));
      arma::mat tmpB = tmp % (arma::ones<arma::vec>(observations.n_rows) *
          trans(condProb.col(i)));

      // Don't update if there's no probability of the Gaussian having points.
      if (probRowSums[i] != 0.0)
      {
        arma::mat covariance = (tmp * trans(tmpB)) / probRowSums[i];
        // Apply covariance constraint.
        constraint.ApplyConstraint(covariance);
        dists[i].Covariance(std::move(covariance));
      }
    }

    // Calculate the new values for omega using the updated conditional
    // probabilities.
    weights = probRowSums / observations.n_cols;

    // Update values of l; calculate new log-likelihood.
    lOld = l;
    l = LogLikelihood(observations, dists, weights);

    iteration++;
  }
}

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy>::Estimate(
    const arma::mat& observations,
    const arma::vec& probabilities,
    std::vector<distribution::GaussianDistribution>& dists,
    arma::vec& weights,
    const bool useInitialModel)
{
  if (!useInitialModel)
    InitialClustering(observations, dists, weights);

  double l = LogLikelihood(observations, dists, weights);

  Log::Debug << "EMFit::Estimate(): initial clustering log-likelihood: "
      << l << std::endl;

  double lOld = -DBL_MAX;
  arma::mat condProb(observations.n_cols, dists.size());

  // Iterate to update the model until no more improvement is found.
  size_t iteration = 1;
  while (std::abs(l - lOld) > tolerance && iteration != maxIterations)
  {
    // Calculate the conditional probabilities of choosing a particular
    // Gaussian given the observations and the present theta value.
    for (size_t i = 0; i < dists.size(); i++)
    {
      // Store conditional probabilities into condProb vector for each
      // Gaussian.  First we make an alias of the condProb vector.
      arma::vec condProbAlias = condProb.unsafe_col(i);
      dists[i].Probability(observations, condProbAlias);
      condProbAlias *= weights[i];
    }

    // Normalize row-wise.
    for (size_t i = 0; i < condProb.n_rows; i++)
    {
      // Avoid dividing by zero; if the probability for everything is 0, we
      // don't want to make it NaN.
      const double probSum = accu(condProb.row(i));
      if (probSum != 0.0)
        condProb.row(i) /= probSum;
    }

    // This will store the sum of probabilities of each state over all the
    // observations.
    arma::vec probRowSums(dists.size());

    // Calculate the new value of the means using the updated conditional
    // probabilities.
    for (size_t i = 0; i < dists.size(); i++)
    {
      // Calculate the sum of probabilities of points, which is the
      // conditional probability of each point being from Gaussian i
      // multiplied by the probability of the point being from this mixture
      // model.
      probRowSums[i] = accu(condProb.col(i) % probabilities);

      dists[i].Mean() = (observations * (condProb.col(i) % probabilities)) /
        probRowSums[i];

      // Calculate the new value of the covariances using the updated
      // conditional probabilities and the updated means.
      arma::mat tmp = observations - (dists[i].Mean() *
          arma::ones<arma::rowvec>(observations.n_cols));
      arma::mat tmpB = tmp % (arma::ones<arma::vec>(observations.n_rows) *
          trans(condProb.col(i) % probabilities));

      arma::mat cov = (tmp * trans(tmpB)) / probRowSums[i];

      // Apply covariance constraint.
      constraint.ApplyConstraint(cov);

      dists[i].Covariance(std::move(cov));
    }

    // Calculate the new values for omega using the updated conditional
    // probabilities.
    weights = probRowSums / accu(probabilities);

    // Update values of l; calculate new log-likelihood.
    lOld = l;
    l = LogLikelihood(observations, dists, weights);

    iteration++;
  }
}

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
void EMFit<InitialClusteringType, CovarianceConstraintPolicy>::
InitialClustering(const arma::mat& observations,
                  std::vector<distribution::GaussianDistribution>& dists,
                  arma::vec& weights)
{
  // Assignments from clustering.
  arma::Row<size_t> assignments;

  // Run clustering algorithm.
  clusterer.Cluster(observations, dists.size(), assignments);

  std::vector<arma::vec> means(dists.size());
  std::vector<arma::mat> covs(dists.size());

  // Now calculate the means, covariances, and weights.
  weights.zeros();
  for (size_t i = 0; i < dists.size(); ++i)
  {
    means[i].zeros(dists[i].Mean().n_elem);
    covs[i].zeros(dists[i].Covariance().n_rows,
                  dists[i].Covariance().n_cols);
  }

  // From the assignments, generate our means, covariances, and weights.
  for (size_t i = 0; i < observations.n_cols; ++i)
  {
    const size_t cluster = assignments[i];

    // Add this to the relevant mean.
    means[cluster] += observations.col(i);

    // Add this to the relevant covariance.
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
    covs[cluster] += normObs * normObs.t();
  }

  for (size_t i = 0; i < dists.size(); ++i)
  {
    covs[i] /= (weights[i] > 1) ? weights[i] : 1;

    // Apply constraints to covariance matrix.
    constraint.ApplyConstraint(covs[i]);

    std::swap(dists[i].Mean(), means[i]);
    dists[i].Covariance(std::move(covs[i]));
  }

  // Finally, normalize weights.
  weights /= accu(weights);
}

template<typename InitialClusteringType, typename CovarianceConstraintPolicy>
double EMFit<InitialClusteringType, CovarianceConstraintPolicy>::LogLikelihood(
    const arma::mat& observations,
    const std::vector<distribution::GaussianDistribution>& dists,
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
void EMFit<InitialClusteringType, CovarianceConstraintPolicy>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(maxIterations, "maxIterations");
  ar & CreateNVP(tolerance, "tolerance");
  ar & CreateNVP(clusterer, "clusterer");
  ar & CreateNVP(constraint, "constraint");
}

} // namespace gmm
} // namespace mlpack

#endif
