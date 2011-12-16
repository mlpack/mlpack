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
  // Determine which Gaussian it will be coming from.
  double gaussRand = math::Random();
  size_t gaussian;

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

  return trans(chol(covariances[gaussian])) *
      arma::randn<arma::vec>(dimensionality) + means[gaussian];
}

void GMM::Estimate(const arma::mat& data)
{
  // Create temporary models and set to the right size.
  std::vector<arma::vec> meansTrial;
  std::vector<arma::mat> covariancesTrial;
  arma::vec weightsTrial;

  arma::mat condProb(data.n_cols, gaussians);

  double l, lOld, bestL, TINY = 1.0e-4;

  bestL = -DBL_MAX;

  KMeans<> k; // Default KMeans parameters, for now.

  // We will perform ten trials, and then save the trial with the best result
  // as our trained model.
  for (size_t iter = 0; iter < 10; iter++)
  {
    InitialClustering(k, data, meansTrial, covariancesTrial, weightsTrial);

    l = Loglikelihood(data, meansTrial, covariancesTrial, weightsTrial);

    Log::Info << "K-means log-likelihood: " << l << std::endl;

    lOld = -DBL_MAX;

    // Iterate to update the model until no more improvement is found.
    size_t maxIterations = 300;
    size_t iteration = 0;
    while (std::abs(l - lOld) > TINY && iteration < maxIterations)
    {
      // Calculate the conditional probabilities of choosing a particular
      // Gaussian given the data and the present theta value.
      for (size_t i = 0; i < gaussians; i++)
      {
        // Store conditional probabilities into condProb vector for each
        // Gaussian.  First we make an alias of the condProb vector.
        arma::vec condProbAlias = condProb.unsafe_col(i);
        phi(data, meansTrial[i], covariancesTrial[i], condProbAlias);
        condProbAlias *= weightsTrial[i];
      }

      // Normalize row-wise.
      for (size_t i = 0; i < condProb.n_rows; i++)
        condProb.row(i) /= accu(condProb.row(i));

      // Store the sum of the probability of each state over all the data.
      arma::vec probRowSums = trans(arma::sum(condProb, 0 /* columnwise */));

      // Calculate the new value of the means using the updated conditional
      // probabilities.
      for (size_t i = 0; i < gaussians; i++)
      {
        meansTrial[i] = (data * condProb.col(i)) / probRowSums[i];

        // Calculate the new value of the covariances using the updated
        // conditional probabilities and the updated means.
        arma::mat tmp = data - (meansTrial[i] *
            arma::ones<arma::rowvec>(data.n_cols));
        arma::mat tmp_b = tmp % (arma::ones<arma::vec>(data.n_rows) *
            trans(condProb.col(i)));

        covariancesTrial[i] = (tmp * trans(tmp_b)) / probRowSums[i];
      }

      // Calculate the new values for omega using the updated conditional
      // probabilities.
      weightsTrial = probRowSums / data.n_cols;

      // Update values of l; calculate new log-likelihood.
      lOld = l;
      l = Loglikelihood(data, meansTrial, covariancesTrial, weightsTrial);

      iteration++;
    }

    Log::Info << "Likelihood of iteration " << iter << " (total " << iteration
        << " iterations): " << l << std::endl;

    // The trial model is trained.  Is it better than our existing model?
    if (l > bestL)
    {
      bestL = l;

      means = meansTrial;
      covariances = covariancesTrial;
      weights = weightsTrial;
    }
  }

  Log::Info << "Log likelihood value of the estimated model: " << bestL << "."
      << std::endl;
  return;
}

/**
 * Estimate the probability distribution directly from the given observations,
 * taking into account the probability of each observation actually being from
 * this distribution.
 */
void GMM::Estimate(const arma::mat& observations,
                   const arma::vec& probabilities)
{
  // This will be very similar to Estimate(const arma::mat&), but there will be
  // minor differences in how we calculate the means, covariances, and weights.
  std::vector<arma::vec> meansTrial;
  std::vector<arma::mat> covariancesTrial;
  arma::vec weightsTrial;

  arma::mat condProb(observations.n_cols, gaussians);

  double l, lOld, bestL, TINY = 1.0e-4;

  bestL = -DBL_MAX;

  KMeans<> k; // Default KMeans parameters, for now.

  // We will perform ten trials, and then save the trial with the best result
  // as our trained model.
  for (size_t iter = 0; iter < 10; iter++)
  {
    InitialClustering(k, observations, meansTrial, covariancesTrial, 
        weightsTrial);

    l = Loglikelihood(observations, meansTrial, covariancesTrial, weightsTrial);

    Log::Info << "K-means log-likelihood: " << l << std::endl;

    lOld = -DBL_MAX;

    // Iterate to update the model until no more improvement is found.
    size_t maxIterations = 300;
    size_t iteration = 0;
    while (std::abs(l - lOld) > TINY && iteration < maxIterations)
    {
      // Calculate the conditional probabilities of choosing a particular
      // Gaussian given the observations and the present theta value.
      for (size_t i = 0; i < gaussians; i++)
      {
        // Store conditional probabilities into condProb vector for each
        // Gaussian.  First we make an alias of the condProb vector.
        arma::vec condProbAlias = condProb.unsafe_col(i);
        phi(observations, meansTrial[i], covariancesTrial[i], condProbAlias);
        condProbAlias *= weightsTrial[i];
      }

      // Normalize row-wise.
      for (size_t i = 0; i < condProb.n_rows; i++)
        condProb.row(i) /= accu(condProb.row(i));

      // This will store the sum of probabilities of each state over all the
      // observations.
      arma::vec probRowSums(gaussians);

      // Calculate the new value of the means using the updated conditional
      // probabilities.
      for (size_t i = 0; i < gaussians; i++)
      {
        // Calculate the sum of probabilities of points, which is the
        // conditional probability of each point being from Gaussian i
        // multiplied by the probability of the point being from this mixture
        // model.
        probRowSums[i] = accu(condProb.col(i) % probabilities);

        meansTrial[i] = (observations * (condProb.col(i) % probabilities)) /
            probRowSums[i];

        // Calculate the new value of the covariances using the updated
        // conditional probabilities and the updated means.
        arma::mat tmp = observations - (meansTrial[i] *
            arma::ones<arma::rowvec>(observations.n_cols));
        arma::mat tmp_b = tmp % (arma::ones<arma::vec>(observations.n_rows) *
            trans(condProb.col(i) % probabilities));

        covariancesTrial[i] = (tmp * trans(tmp_b)) / probRowSums[i];
      }

      // Calculate the new values for omega using the updated conditional
      // probabilities.
      weightsTrial = probRowSums / accu(probabilities);

      // Update values of l; calculate new log-likelihood.
      lOld = l;
      l = Loglikelihood(observations, meansTrial, 
                        covariancesTrial, weightsTrial);

      iteration++;
    }

    Log::Info << "Likelihood of iteration " << iter << " (total " << iteration
        << " iterations): " << l << std::endl;

    // The trial model is trained.  Is it better than our existing model?
    if (l > bestL)
    {
      bestL = l;

      means = meansTrial;
      covariances = covariancesTrial;
      weights = weightsTrial;
    }
  }

  Log::Info << "Log likelihood value of the estimated model: " << bestL << "."
      << std::endl;
  return;
}

double GMM::Loglikelihood(const arma::mat& data,
                          const std::vector<arma::vec>& meansL,
                          const std::vector<arma::mat>& covariancesL,
                          const arma::vec& weightsL) const
{
  double loglikelihood = 0;

  arma::vec phis;
  arma::mat likelihoods(gaussians, data.n_cols);
  for (size_t i = 0; i < gaussians; i++)
  {
    phi(data, meansL[i], covariancesL[i], phis);
    likelihoods.row(i) = weightsL(i) * trans(phis);
  }

  // Now sum over every point.
  for (size_t j = 0; j < data.n_cols; j++)
    loglikelihood += log(accu(likelihoods.col(j)));

  return loglikelihood;
}
