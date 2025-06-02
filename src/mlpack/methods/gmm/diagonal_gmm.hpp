/**
 * @author Kim SangYeon
 * @file methods/gmm/diagonal_gmm.hpp
 *
 * Defines a Diagonal Gaussian Mixture model and estimates the parameters
 * of the model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GMM_DIAGONAL_GMM_HPP
#define MLPACK_METHODS_GMM_DIAGONAL_GMM_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distributions/diagonal_gaussian_distribution.hpp>
#include <mlpack/core/math/log_add.hpp>

// This is the default fitting method class.
#include "em_fit.hpp"

// This is the default covariance matrix constraint.
#include "diagonal_constraint.hpp"

namespace mlpack {

/**
 * A Diagonal Gaussian Mixture Model. 
 * This class uses maximum likelihood loss functions to estimate the parameters
 * of the DiagonalGMM on a given dataset via the given fitting mechanism, 
 * defined by the FittingType template parameter.  The DiagonalGMM can be
 * trained using normal data, or data with probabilities of being
 * from this GMM (see DiagonalGMM::Train() for more information).
 * The DiagonalGMM is the same as GMM except for wrapping gmm_diag class.
 *
 * The Train() method uses a template type 'FittingType'.  The FittingType
 * template class must provide a way for the DiagonalGMM to train on data.
 * It must provide the following two functions:
 *
 * @code
 * void Estimate(
 *     const arma::mat& observations,
 *     std::vector<DiagonalGaussianDistribution<>>& dists,
 *     arma::vec& weights);
 *
 * void Estimate(
 *     const arma::mat& observations,
 *     const arma::vec& probabilities,
 *     std::vector<DiagonalGaussianDistribution<>>& dists,
 *     arma::vec& weights);
 * @endcode
 *
 * Example use:
 *
 * @code
 * // Set up a mixture of 5 gaussians in a 4-dimensional space.
 * DiagonalGMM g(5, 4);
 *
 * // Train the DiagonalGMM given the data observations, using the default
 * // EM fitting mechanism.
 * 
 * g.Train(data);
 *
 * // Get the probability of 'observation' being observed from this
 * // DiagoanlGMM.
 * double probability = g.Probability(observation);
 *
 * // Get a random observation from the DiagonalGMM.
 * arma::vec observation = g.Random();
 * @endcode
 */
class DiagonalGMM
{
 private:
  //! The number of Gaussians in the model.
  size_t gaussians;
  //! The dimensionality of the model.
  size_t dimensionality;

  //! Vector of Gaussians.
  std::vector<DiagonalGaussianDistribution<>> dists;

  //! Vector of a priori weights for each Gaussian.
  arma::vec weights;

 public:
  /**
   * Create an empty Diagonal Gaussian Mixture Model, with zero gaussians.
   */
  DiagonalGMM() :
      gaussians(0),
      dimensionality(0)
  {
    // Warn the user.  They probably don't want to do this.  If this
    // constructor is being used (because it is required by some template
    // classes), the user should know that it is potentially dangerous.
    Log::Debug << "DiagonalGMM::DiagonalGMM(): no parameters given;"
        "Estimate() may fail " << "unless parameters are set." << std::endl;
  }

  /**
   * Create a GMM with the given number of Gaussians, each of which have the
   * specified dimensionality.  The means and covariances will be set to 0.
   *
   * @param gaussians Number of Gaussians in this DiagonalGMM.
   * @param dimensionality Dimensionality of each Gaussian.
   */
  DiagonalGMM(const size_t gaussians, const size_t dimensionality);

  /**
   * Create a DiagonalGMM with the given dists and weights.
   *
   * @param dists Distributions of the model.
   * @param weights Weights of the model.
   */
  DiagonalGMM(const std::vector<DiagonalGaussianDistribution<>>& dists,
              const arma::vec& weights) :
      gaussians(dists.size()),
      dimensionality((!dists.empty()) ? dists[0].Mean().n_elem : 0),
      dists(dists),
      weights(weights) { /* Nothing to do. */ }

  //! Copy constructor for DiagonalGMMs.
  DiagonalGMM(const DiagonalGMM& other);

  //! Copy operator for DiagonalGMMs.
  DiagonalGMM& operator=(const DiagonalGMM& other);

  //! Return the number of Gaussians in the model.
  size_t Gaussians() const { return gaussians; }
  //! Return the dimensionality of the model.
  size_t Dimensionality() const { return dimensionality; }

  /**
   * Return a const reference to a component distribution.
   *
   * @param i Index of component.
   */
  const DiagonalGaussianDistribution<>& Component(size_t i) const
  {
    return dists[i];
  }

  /**
   * Return a reference to a component distribution.
   *
   * @param i Index of component.
   */
  DiagonalGaussianDistribution<>& Component(size_t i)
  {
    return dists[i];
  }

  //! Return a const reference to the a priori weights of each Gaussian.
  const arma::vec& Weights() const { return weights; }
  //! Return a reference to the a priori weights of each Gaussian.
  arma::vec& Weights() { return weights; }

  /**
   * Return the probability that the given observation came from this
   * distribution.
   *
   * @param observation Observation to evaluate the probability of.
   */
  double Probability(const arma::vec& observation) const;

  /**
   * Return the probability that the given observation matrix.
   *
   * @param observation Observation to evaluate the probability of.
   * @param probs Stores the value of probability for observation.
   */
  void Probability(const arma::mat& observation, arma::vec& probs) const;

  /**
   * Return the log probability that the given observation came from this
   * distribution.
   *
   * @param observation Observation to evaluate the log-probability of.
   */
  double LogProbability(const arma::vec& observation) const;

  /**
   * Return the log probability that the given observation matrix.
   *
   * @param observation Observation to evaluate the log-probability of.
   * @param logProbs Stores the value of log-probability for observation.
   */
  void LogProbability(const arma::mat& observation, arma::vec& logProbs) const;

  /**
   * Return the probability that the given observation came from the given
   * Gaussian component in this distribution.
   *
   * @param observation Observation to evaluate the probability of.
   * @param component Index of the component of the DiagonalGMM.
   */
  double Probability(const arma::vec& observation,
                     const size_t component) const;

  /**
   * Return the log probability that the given observation came from the given
   * Gaussian component in this distribution.
   *
   * @param observation Observation to evaluate the probability of.
   * @param component Index of the component of the DiagonalGMM.
   */
  double LogProbability(const arma::vec& observation,
                        const size_t component) const;
  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this DiagonalGMM.
   */
  arma::vec Random() const;

  /**
   * Estimate the probability distribution directly from the given
   * observations, using the given algorithm in the FittingType class to fit
   * the data.
   *
   * The fitting will be performed 'trials' times; from these trials, the model
   * with the greatest log-likelihood will be selected.  By default, only one
   * trial is performed.  The log-likelihood of the best fitting is returned.
   *
   * Optionally, the existing model can be used as an initial model for the
   * estimation by setting 'useExistingModel' to true.  If the fitting
   * procedure is deterministic after the initial position is given, then
   * 'trials' should be set to 1.
   *
   * @param observations Observations of the model.
   * @param trials Number of trials to perform; the model in these trials with
   *     the greatest log-likelihood will be selected.
   * @param useExistingModel If true, the existing model is used as an initial
   *     model for the estimation.
   * @param fitter Fitting type that estimates observations.
   * @return The log-likelihood of the best fit.
   */
  template<typename FittingType = EMFit<KMeans<>, DiagonalConstraint,
      DiagonalGaussianDistribution<>>>
  double Train(const arma::mat& observations,
               const size_t trials = 1,
               const bool useExistingModel = false,
               FittingType fitter = FittingType());

  /**
   * Estimate the probability distribution directly from the given observations,
   * taking into account the probability of each observation actually being from
   * this distribution, and using the given algorithm in the FittingType class
   * to fit the data.
   *
   * The fitting will be performed 'trials' times; from these trials, the model
   * with the greatest log-likelihood will be selected.  By default, only one
   * trial is performed.  The log-likelihood of the best fitting is returned.
   *
   * Optionally, the existing model can be used as an initial model for the
   * estimation by setting 'useExistingModel' to true.  If the fitting procedure
   * is deterministic after the initial position is given, then 'trials' should
   * be set to 1.
   *
   * @param observations Observations of the model.
   * @param probabilities Probability of each observation being from this
   *     distribution.
   * @param trials Number of trials to perform; the model in these trials with
   *     the greatest log-likelihood will be selected.
   * @param useExistingModel If true, the existing model is used as an initial
   *     model for the estimation.
   * @param fitter Fitting type that estimates observations.
   * @return The log-likelihood of the best fit.
   */
  template<typename FittingType = EMFit<KMeans<>, DiagonalConstraint,
      DiagonalGaussianDistribution<>>>
  double Train(const arma::mat& observations,
               const arma::vec& probabilities,
               const size_t trials = 1,
               const bool useExistingModel = false,
               FittingType fitter = FittingType());

  /**
   * Classify the given observations as being from an individual component in
   * this DiagonalGMM. The resultant classifications are stored in the 'labels'
   * object, and each label will be between 0 and (Gaussians() - 1). Supposing
   * that a point was classified with label 2, and that our DiagonalGMM object
   * was called 'dgmm', one could access the relevant Gaussian distribution as
   * follows:
   *
   * @code
   * arma::vec mean = dgmm.Means()[2];
   * arma::mat covariance = dgmm.Covariances()[2];
   * double priorWeight = dgmm.Weights()[2];
   * @endcode
   *
   * @param observations Matrix of observations to classify.
   * @param labels Object which will be filled with labels.
   */
  void Classify(const arma::mat& observations,
                arma::Row<size_t>& labels) const;

  /**
   * Serialize the DiagonalGMM.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * This function computes the log-likelihood of the given model and is used
   * by DiagonalGMM::Train().
   *
   * @param observations Matrix of observations.
   * @param means Means of the given mixture model.
   * @param covariances Covariances of the given mixture model.
   * @param weights Weights of the given mixture model.
   */
  double LogLikelihood(
      const arma::mat& observations,
      const std::vector<DiagonalGaussianDistribution<>>& dists,
      const arma::vec& weights) const;
};

} // namespace mlpack

// Include implementation.
#include "diagonal_gmm_impl.hpp"

#endif // MLPACK_METHODS_GMM_DIAGONAL_GMM_HPP
