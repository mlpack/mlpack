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
#include "kmeans.hpp"

using namespace mlpack;
using namespace gmm;

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

  // We will perform ten trials, and then save the trial with the best result
  // as our trained model.
  for (size_t iter = 0; iter < 10; iter++)
  {
    // Use k-means to find initial values for the parameters.
    KMeans(data, gaussians, means_trial, covariances_trial, weights_trial);

    // Calculate the log likelihood of the model.
    l = Loglikelihood(data, means_trial, covariances_trial, weights_trial);

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
      arma::vec prob_row_sums = arma::sum(cond_prob, 0 /* column-wise */);

      // Calculate the new value of the means using the updated conditional
      // probabilities.
      for (size_t i = 0; i < gaussians; i++)
      {
        means_trial[i].zeros();
        for (size_t j = 0; j < data.n_cols; j++)
          means_trial[i] += cond_prob(j, i) * data.col(j);

        means_trial[i] /= prob_row_sums[i];
      }

      // Calculate the new value of the covariances using the updated
      // conditional probabilities and the updated means.
      for (size_t i = 0; i < gaussians; i++)
      {
        covariances_trial[i].zeros();
        for (size_t j = 0; j < data.n_cols; j++)
        {
          arma::vec tmp = data.col(j) - means_trial[i];
          covariances_trial[i] += cond_prob(j, i) * (tmp * trans(tmp));
        }

        covariances_trial[i] /= prob_row_sums[i];
      }

      // Calculate the new values for omega using the updated conditional
      // probabilities.
      weights_trial = prob_row_sums / data.n_cols;

      // Update values of l; calculate new log-likelihood.
      l_old = l;
      l = Loglikelihood(data, means_trial, covariances_trial, weights_trial);

      iteration++;
    }

    Log::Warn << "Likelihood of iteration " << iter << " (total " << iteration
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
  long double likelihood;

  for (size_t j = 0; j < data.n_cols; j++)
  {
    likelihood = 0;
    for(size_t i = 0; i < gaussians; i++)
      likelihood += weights_l(i) * phi(data.unsafe_col(j), means_l[i],
          covariances_l[i]);

    loglikelihood += log(likelihood);
  }

  return loglikelihood;
}
