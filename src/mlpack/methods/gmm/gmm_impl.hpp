/**
 * @file gmm_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * Implementation of template-based GMM methods.
 *
 * This file is part of MLPACK 1.0.11.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_GMM_GMM_IMPL_HPP
#define __MLPACK_METHODS_GMM_GMM_IMPL_HPP

// In case it hasn't already been included.
#include "gmm.hpp"

#include <mlpack/core/util/save_restore_utility.hpp>

namespace mlpack {
namespace gmm {

/**
 * Create a GMM with the given number of Gaussians, each of which have the
 * specified dimensionality.  The means and covariances will be set to 0.
 *
 * @param gaussians Number of Gaussians in this GMM.
 * @param dimensionality Dimensionality of each Gaussian.
 */
template<typename FittingType>
GMM<FittingType>::GMM(const size_t gaussians, const size_t dimensionality) :
    gaussians(gaussians),
    dimensionality(dimensionality),
    means(gaussians, arma::vec(dimensionality)),
    covariances(gaussians, arma::mat(dimensionality, dimensionality)),
    weights(gaussians),
    localFitter(FittingType()),
    fitter(localFitter)
{
  // Clear the memory; set it to 0.  Technically this model is still valid, but
  // only barely.
  weights.fill(1.0 / gaussians);
  for (size_t i = 0; i < gaussians; ++i)
  {
    means[i].zeros();
    covariances[i].eye();
  }
}

/**
 * Create a GMM with the given number of Gaussians, each of which have the
 * specified dimensionality.  Also, pass in an initialized FittingType class;
 * this is useful in cases where the FittingType class needs to store some
 * state.
 *
 * @param gaussians Number of Gaussians in this GMM.
 * @param dimensionality Dimensionality of each Gaussian.
 * @param fitter Initialized fitting mechanism.
 */
template<typename FittingType>
GMM<FittingType>::GMM(const size_t gaussians,
                      const size_t dimensionality,
                      FittingType& fitter) :
    gaussians(gaussians),
    dimensionality(dimensionality),
    means(gaussians, arma::vec(dimensionality)),
    covariances(gaussians, arma::mat(dimensionality, dimensionality)),
    weights(gaussians),
    fitter(fitter)
{
  // Clear the memory; set it to 0.  Technically this model is still valid, but
  // only barely.
  weights.fill(1.0 / gaussians);
  for (size_t i = 0; i < gaussians; ++i)
  {
    means[i].zeros();
    covariances[i].eye();
  }
}

// Copy constructor.
template<typename FittingType>
template<typename OtherFittingType>
GMM<FittingType>::GMM(const GMM<OtherFittingType>& other) :
    gaussians(other.Gaussians()),
    dimensionality(other.Dimensionality()),
    means(other.Means()),
    covariances(other.Covariances()),
    weights(other.Weights()),
    localFitter(FittingType()),
    fitter(localFitter) { /* Nothing to do. */ }

// Copy constructor for when the other GMM uses the same fitting type.
template<typename FittingType>
GMM<FittingType>::GMM(const GMM<FittingType>& other) :
    gaussians(other.Gaussians()),
    dimensionality(other.Dimensionality()),
    means(other.Means()),
    covariances(other.Covariances()),
    weights(other.Weights()),
    localFitter(other.Fitter()),
    fitter(localFitter) { /* Nothing to do. */ }

template<typename FittingType>
template<typename OtherFittingType>
GMM<FittingType>& GMM<FittingType>::operator=(
    const GMM<OtherFittingType>& other)
{
  gaussians = other.Gaussians();
  dimensionality = other.Dimensionality();
  means = other.Means();
  covariances = other.Covariances();
  weights = other.Weights();

  return *this;
}

template<typename FittingType>
GMM<FittingType>& GMM<FittingType>::operator=(const GMM<FittingType>& other)
{
  gaussians = other.Gaussians();
  dimensionality = other.Dimensionality();
  means = other.Means();
  covariances = other.Covariances();
  weights = other.Weights();
  localFitter = other.Fitter();

  return *this;
}

// Load a GMM from file.
template<typename FittingType>
void GMM<FittingType>::Load(const std::string& filename)
{
  util::SaveRestoreUtility load;

  if (!load.ReadFile(filename))
    Log::Fatal << "GMM::Load(): could not read file '" << filename << "'!\n";

  load.LoadParameter(gaussians, "gaussians");
  load.LoadParameter(dimensionality, "dimensionality");
  load.LoadParameter(weights, "weights");

  // We need to do a little error checking here.
  if (weights.n_elem != gaussians)
  {
    Log::Fatal << "GMM::Load('" << filename << "'): file reports " << gaussians
        << " gaussians but weights vector only contains " << weights.n_elem
        << " elements!" << std::endl;
  }

  means.resize(gaussians);
  covariances.resize(gaussians);

  for (size_t i = 0; i < gaussians; ++i)
  {
    std::stringstream o;
    o << i;
    std::string meanName = "mean" + o.str();
    std::string covName = "covariance" + o.str();

    load.LoadParameter(means[i], meanName);
    load.LoadParameter(covariances[i], covName);
  }
}

// Save a GMM to a file.
template<typename FittingType>
void GMM<FittingType>::Save(const std::string& filename) const
{
  util::SaveRestoreUtility save;
  save.SaveParameter(gaussians, "gaussians");
  save.SaveParameter(dimensionality, "dimensionality");
  save.SaveParameter(weights, "weights");
  for (size_t i = 0; i < gaussians; ++i)
  {
    // Generate names for the XML nodes.
    std::stringstream o;
    o << i;
    std::string meanName = "mean" + o.str();
    std::string covName = "covariance" + o.str();

    // Now save them.
    save.SaveParameter(means[i], meanName);
    save.SaveParameter(covariances[i], covName);
  }

  if (!save.WriteFile(filename))
    Log::Warn << "GMM::Save(): error saving to '" << filename << "'.\n";
}

/**
 * Return the probability of the given observation being from this GMM.
 */
template<typename FittingType>
double GMM<FittingType>::Probability(const arma::vec& observation) const
{
  // Sum the probability for each Gaussian in our mixture (and we have to
  // multiply by the prior for each Gaussian too).
  double sum = 0;
  for (size_t i = 0; i < gaussians; i++)
    sum += weights[i] * phi(observation, means[i], covariances[i]);

  return sum;
}

/**
 * Return the probability of the given observation being from the given
 * component in the mixture.
 */
template<typename FittingType>
double GMM<FittingType>::Probability(const arma::vec& observation,
                                     const size_t component) const
{
  // We are only considering one Gaussian component -- so we only need to call
  // phi() once.  We do consider the prior probability!
  return weights[component] *
      phi(observation, means[component], covariances[component]);
}

/**
 * Return a randomly generated observation according to the probability
 * distribution defined by this object.
 */
template<typename FittingType>
arma::vec GMM<FittingType>::Random() const
{
  // Determine which Gaussian it will be coming from.
  double gaussRand = math::Random();
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

  return trans(chol(covariances[gaussian])) *
      arma::randn<arma::vec>(dimensionality) + means[gaussian];
}

/**
 * Fit the GMM to the given observations.
 */
template<typename FittingType>
double GMM<FittingType>::Estimate(const arma::mat& observations,
                                  const size_t trials,
                                  const bool useExistingModel)
{
  double bestLikelihood; // This will be reported later.

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the GMM was
    // initialized with no parameters (0 gaussians, dimensionality of 0).
    fitter.Estimate(observations, means, covariances, weights,
        useExistingModel);

    bestLikelihood = LogLikelihood(observations, means, covariances, weights);
  }
  else
  {
    if (trials == 0)
      return -DBL_MAX; // It's what they asked for...

    // If each trial must start from the same initial location, we must save it.
    std::vector<arma::vec> meansOrig;
    std::vector<arma::mat> covariancesOrig;
    arma::vec weightsOrig;
    if (useExistingModel)
    {
      meansOrig = means;
      covariancesOrig = covariances;
      weightsOrig = weights;
    }

    // We need to keep temporary copies.  We'll do the first training into the
    // actual model position, so that if it's the best we don't need to copy it.
    fitter.Estimate(observations, means, covariances, weights,
        useExistingModel);

    bestLikelihood = LogLikelihood(observations, means, covariances, weights);

    Log::Info << "GMM::Estimate(): Log-likelihood of trial 0 is "
        << bestLikelihood << "." << std::endl;

    // Now the temporary model.
    std::vector<arma::vec> meansTrial(gaussians, arma::vec(dimensionality));
    std::vector<arma::mat> covariancesTrial(gaussians,
        arma::mat(dimensionality, dimensionality));
    arma::vec weightsTrial(gaussians);

    for (size_t trial = 1; trial < trials; ++trial)
    {
      if (useExistingModel)
      {
        meansTrial = meansOrig;
        covariancesTrial = covariancesOrig;
        weightsTrial = weightsOrig;
      }

      fitter.Estimate(observations, meansTrial, covariancesTrial, weightsTrial,
          useExistingModel);

      // Check to see if the log-likelihood of this one is better.
      double newLikelihood = LogLikelihood(observations, meansTrial,
          covariancesTrial, weightsTrial);

      Log::Info << "GMM::Estimate(): Log-likelihood of trial " << trial
          << " is " << newLikelihood << "." << std::endl;

      if (newLikelihood > bestLikelihood)
      {
        // Save new likelihood and copy new model.
        bestLikelihood = newLikelihood;

        means = meansTrial;
        covariances = covariancesTrial;
        weights = weightsTrial;
      }
    }
  }

  // Report final log-likelihood and return it.
  Log::Info << "GMM::Estimate(): log-likelihood of trained GMM is "
      << bestLikelihood << "." << std::endl;
  return bestLikelihood;
}

/**
 * Fit the GMM to the given observations, each of which has a certain
 * probability of being from this distribution.
 */
template<typename FittingType>
double GMM<FittingType>::Estimate(const arma::mat& observations,
                                  const arma::vec& probabilities,
                                  const size_t trials,
                                  const bool useExistingModel)
{
  double bestLikelihood; // This will be reported later.

  // We don't need to store temporary models if we are only doing one trial.
  if (trials == 1)
  {
    // Train the model.  The user will have been warned earlier if the GMM was
    // initialized with no parameters (0 gaussians, dimensionality of 0).
    fitter.Estimate(observations, probabilities, means, covariances, weights,
        useExistingModel);

    bestLikelihood = LogLikelihood(observations, means, covariances, weights);
  }
  else
  {
    if (trials == 0)
      return -DBL_MAX; // It's what they asked for...

    // If each trial must start from the same initial location, we must save it.
    std::vector<arma::vec> meansOrig;
    std::vector<arma::mat> covariancesOrig;
    arma::vec weightsOrig;
    if (useExistingModel)
    {
      meansOrig = means;
      covariancesOrig = covariances;
      weightsOrig = weights;
    }

    // We need to keep temporary copies.  We'll do the first training into the
    // actual model position, so that if it's the best we don't need to copy it.
    fitter.Estimate(observations, probabilities, means, covariances, weights,
        useExistingModel);

    bestLikelihood = LogLikelihood(observations, means, covariances, weights);

    Log::Debug << "GMM::Estimate(): Log-likelihood of trial 0 is "
        << bestLikelihood << "." << std::endl;

    // Now the temporary model.
    std::vector<arma::vec> meansTrial(gaussians, arma::vec(dimensionality));
    std::vector<arma::mat> covariancesTrial(gaussians,
        arma::mat(dimensionality, dimensionality));
    arma::vec weightsTrial(gaussians);

    for (size_t trial = 1; trial < trials; ++trial)
    {
      if (useExistingModel)
      {
        meansTrial = meansOrig;
        covariancesTrial = covariancesOrig;
        weightsTrial = weightsOrig;
      }

      fitter.Estimate(observations, meansTrial, covariancesTrial, weightsTrial,
          useExistingModel);

      // Check to see if the log-likelihood of this one is better.
      double newLikelihood = LogLikelihood(observations, meansTrial,
          covariancesTrial, weightsTrial);

      Log::Debug << "GMM::Estimate(): Log-likelihood of trial " << trial
          << " is " << newLikelihood << "." << std::endl;

      if (newLikelihood > bestLikelihood)
      {
        // Save new likelihood and copy new model.
        bestLikelihood = newLikelihood;

        means = meansTrial;
        covariances = covariancesTrial;
        weights = weightsTrial;
      }
    }
  }

  // Report final log-likelihood and return it.
  Log::Info << "GMM::Estimate(): log-likelihood of trained GMM is "
      << bestLikelihood << "." << std::endl;
  return bestLikelihood;
}

/**
 * Classify the given observations as being from an individual component in this
 * GMM.
 */
template<typename FittingType>
void GMM<FittingType>::Classify(const arma::mat& observations,
                                arma::Col<size_t>& labels) const
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
template<typename FittingType>
double GMM<FittingType>::LogLikelihood(
    const arma::mat& data,
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

template<typename FittingType>
std::string GMM<FittingType>::ToString() const
{
  std::ostringstream convert;
  std::ostringstream data;
  convert << "GMM [" << this << "]" << std::endl;
  convert << "  Gaussians: " << gaussians << std::endl;
  convert << "  Dimensionality: "<<dimensionality;
  convert << std::endl;
  // Secondary ostringstream so things can be indented properly.
  for (size_t ind=0; ind < gaussians; ind++)
  {
    data << "Means of Gaussian " << ind << ": " << std::endl << means[ind]; 
    data << std::endl;
    data << "Covariances of Gaussian " << ind << ": " << std::endl ;
    data << covariances[ind] << std::endl;
    data << "Weight of Gaussian " << ind << ": " << std::endl ;
    data << weights[ind] << std::endl;
  }

  convert << util::Indent(data.str());

  return convert.str();
}

}; // namespace gmm
}; // namespace mlpack

#endif

