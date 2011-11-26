/**
 * @file gmm.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * Implementation for the loglikelihood function, the EM algorithm
 * and also computes the K-means for getting an initial point
 */
#include "gmm.hpp"
#include "phi.hpp"

#include <mlpack/methods/kmeans/kmeans.hpp>

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::kmeans;

double GMM::Probability(const arma::vec& observation) const
{
  // Sum the probability for each Gaussian in our mixture (and we have to
  // multiply by the prior for each Gaussian too).
  double sum = 0;
  for (size_t i = 0; i < gaussians; i++)
    sum += weights[i] * phi(observation, means[i], covariances[i]);

  return sum;
}

/**
 * Return a randomly generated observation according to the probability
 * distribution defined by this object.
 */
arma::vec GMM::Random() const
{
  return "0 0";
}

void GMM::ExpectationMaximization(const arma::mat& data)
{
  // Create temporary models and set to the right size.
  std::vector<arma::vec> means_trial(gaussians, arma::vec(dimension));
  std::vector<arma::mat> covariances_trial(gaussians,
      arma::mat(dimension, dimension));
  arma::vec weights_trial(gaussians);

  arma::mat cond_prob(data.n_cols, gaussians);

  long double l, l_old, best_l, TINY = 1.0e-4;

  best_l = -DBL_MAX;

  KMeans k; // Default KMeans parameters, for now.

  // We will perform ten trials, and then save the trial with the best result
  // as our trained model.
  for (size_t iter = 0; iter < 10; iter++)
  {
    arma::Col<size_t> assignments;

    k.Cluster(data, gaussians, assignments);

    // Clear the weights, covariances, and means, before we recalculate them.
    weights_trial.zeros();
    for (size_t i = 0; i < gaussians; i++)
    {
      means_trial[i].zeros();
      covariances_trial[i].zeros();
    }

    // From the assignments, generate our means, covariances, and weights.
    for (size_t i = 0; i < data.n_cols; i++)
    {
      size_t cluster = assignments[i];

      // Add this to the relevant mean.
      means_trial[cluster] += data.col(i);

      // And add it to the relative covariance matrix.
      covariances_trial[cluster] += data.col(i) * trans(data.col(i));

      // Now add one to the weights (we will normalize).
      weights_trial[cluster]++;
    }

    // Now normalize the means, covariances, and weights.
    for (size_t i = 0; i < gaussians; i++)
    {
      // Normalize mean.
      means_trial[i] /= weights_trial[i];

      // Normalize covariance (use unbiased estimator).
      covariances_trial[i] /= (weights_trial[i] - 1);
    }

    // Finally, normalize weights.
    weights_trial /= accu(weights_trial);

    l = Loglikelihood(data, means_trial, covariances_trial, weights_trial);

    Log::Info << "K-means log-likelihood: " << l << std::endl;

    l_old = -DBL_MAX;

    // Iterate to update the model until no more improvement is found.
    size_t max_iterations = 300;
    size_t iteration = 0;
    while (std::abs(l - l_old) > TINY && iteration < max_iterations)
    {
      // Calculate the conditional probabilities of choosing a particular
      // Gaussian given the data and the present theta value.
      for (size_t i = 0; i < gaussians; i++)
      {
        // Store conditional probabilities into cond_prob vector for each
        // Gaussian.  First we make an alias of the cond_prob vector.
        arma::vec cond_prob_alias = cond_prob.unsafe_col(i);
        phi(data, means_trial[i], covariances_trial[i], cond_prob_alias);
        cond_prob_alias *= weights_trial[i];
      }

      // Normalize row-wise.
      for (size_t i = 0; i < cond_prob.n_rows; i++)
        cond_prob.row(i) /= accu(cond_prob.row(i));

      // Store the sum of the probability of each state over all the data.
      arma::vec prob_row_sums = trans(arma::sum(cond_prob, 0 /* columnwise */));

      // Calculate the new value of the means using the updated conditional
      // probabilities.
      for (size_t i = 0; i < gaussians; i++)
      {
        means_trial[i] = (data * cond_prob.col(i)) / prob_row_sums[i];

        // Calculate the new value of the covariances using the updated
        // conditional probabilities and the updated means.
        arma::mat tmp = data - (means_trial[i] *
            arma::ones<arma::rowvec>(data.n_cols));
        arma::mat tmp_b = tmp % (arma::ones<arma::vec>(data.n_rows) *
            trans(cond_prob.col(i)));

        covariances_trial[i] = (tmp * trans(tmp_b)) / prob_row_sums[i];
      }

      // Calculate the new values for omega using the updated conditional
      // probabilities.
      weights_trial = prob_row_sums / data.n_cols;

      // Update values of l; calculate new log-likelihood.
      l_old = l;
      l = Loglikelihood(data, means_trial, covariances_trial, weights_trial);

      iteration++;
    }

    Log::Info << "Likelihood of iteration " << iter << " (total " << iteration
        << " iterations): " << l << std::endl;

    // The trial model is trained.  Is it better than our existing model?
    if (l > best_l)
    {
      best_l = l;

      means = means_trial;
      covariances = covariances_trial;
      weights = weights_trial;
    }
  }

  Log::Info << "Log likelihood value of the estimated model: " << best_l << "."
      << std::endl;
  return;
}

long double GMM::Loglikelihood(const arma::mat& data,
                               const std::vector<arma::vec>& means_l,
                               const std::vector<arma::mat>& covariances_l,
                               const arma::vec& weights_l) const
{
  long double loglikelihood = 0;

  arma::vec phis;
  arma::mat likelihoods(gaussians, data.n_cols);
  for (size_t i = 0; i < gaussians; i++)
  {
    phi(data, means_l[i], covariances_l[i], phis);
    likelihoods.row(i) = weights_l(i) * trans(phis);
  }

  // Now sum over every point.
  for (size_t j = 0; j < data.n_cols; j++)
    loglikelihood += log(accu(likelihoods.col(j)));

  return loglikelihood;
}
