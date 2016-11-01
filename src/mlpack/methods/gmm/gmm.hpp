/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Michael Fox
 * @file gmm.hpp
 *
 * Defines a Gaussian Mixture model and
 * estimates the parameters of the model
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MOG_MOG_EM_HPP
#define MLPACK_METHODS_MOG_MOG_EM_HPP

#include <mlpack/core.hpp>

// This is the default fitting method class.
#include "em_fit.hpp"

namespace mlpack {
namespace gmm /** Gaussian Mixture Models. */ {

/**
 * A Gaussian Mixture Model (GMM). This class uses maximum likelihood loss
 * functions to estimate the parameters of the GMM on a given dataset via the
 * given fitting mechanism, defined by the FittingType template parameter.  The
 * GMM can be trained using normal data, or data with probabilities of being
 * from this GMM (see GMM::Train() for more information).
 *
 * The Train() method uses a template type 'FittingType'.  The FittingType
 * template class must provide a way for the GMM to train on data.  It must
 * provide the following two functions:
 *
 * @code
 * void Estimate(const arma::mat& observations,
 *               std::vector<distribution::GaussianDistribution>& dists,
 *               arma::vec& weights);
 *
 * void Estimate(const arma::mat& observations,
 *               const arma::vec& probabilities,
 *               std::vector<distribution::GaussianDistribution>& dists,
 *               arma::vec& weights);
 * @endcode
 *
 * These functions should produce a trained GMM from the given observations and
 * probabilities.  These may modify the size of the model (by increasing the
 * size of the mean and covariance vectors as well as the weight vectors), but
 * the method should expect that these vectors are already set to the size of
 * the GMM as specified in the constructor.
 *
 * For a sample implementation, see the EMFit class; this class uses the EM
 * algorithm to train a GMM, and is the default fitting type for the Train()
 * method.
 *
 * The GMM, once trained, can be used to generate random points from the
 * distribution and estimate the probability of points being from the
 * distribution.  The parameters of the GMM can be obtained through the
 * accessors and mutators.
 *
 * Example use:
 *
 * @code
 * // Set up a mixture of 5 gaussians in a 4-dimensional space.
 * GMM g(5, 4);
 *
 * // Train the GMM given the data observations, using the default EM fitting
 * // mechanism.
 * g.Train(data);
 *
 * // Get the probability of 'observation' being observed from this GMM.
 * double probability = g.Probability(observation);
 *
 * // Get a random observation from the GMM.
 * arma::vec observation = g.Random();
 * @endcode
 */
class GMM
{
 private:
  //! The number of Gaussians in the model.
  size_t gaussians;
  //! The dimensionality of the model.
  size_t dimensionality;

  //! Vector of Gaussians
  std::vector<distribution::GaussianDistribution> dists;

  //! Vector of a priori weights for each Gaussian.
  arma::vec weights;

 public:
  /**
   * Create an empty Gaussian Mixture Model, with zero gaussians.
   */
  GMM() :
      gaussians(0),
      dimensionality(0)
  {
    // Warn the user.  They probably don't want to do this.  If this constructor
    // is being used (because it is required by some template classes), the user
    // should know that it is potentially dangerous.
    Log::Debug << "GMM::GMM(): no parameters given; Estimate() may fail "
        << "unless parameters are set." << std::endl;
  }

  /**
   * Create a GMM with the given number of Gaussians, each of which have the
   * specified dimensionality.  The means and covariances will be set to 0.
   *
   * @param gaussians Number of Gaussians in this GMM.
   * @param dimensionality Dimensionality of each Gaussian.
   */
  GMM(const size_t gaussians, const size_t dimensionality);

  /**
   * Create a GMM with the given dists and weights.
   *
   * @param dists Distributions of the model.
   * @param weights Weights of the model.
   */
  GMM(const std::vector<distribution::GaussianDistribution> & dists,
      const arma::vec& weights) :
      gaussians(dists.size()),
      dimensionality((!dists.empty()) ? dists[0].Mean().n_elem : 0),
      dists(dists),
      weights(weights) { /* Nothing to do. */ }

  //! Copy constructor for GMMs.
  GMM(const GMM& other);

  //! Copy operator for GMMs.
  GMM& operator=(const GMM& other);

  //! Return the number of gaussians in the model.
  size_t Gaussians() const { return gaussians; }
  //! Return the dimensionality of the model.
  size_t Dimensionality() const { return dimensionality; }

  /**
   * Return a const reference to a component distribution.
   *
   * @param i index of component.
   */
  const distribution::GaussianDistribution& Component(size_t i) const {
      return dists[i]; }
  /**
   * Return a reference to a component distribution.
   *
   * @param i index of component.
   */
  distribution::GaussianDistribution& Component(size_t i) { return dists[i]; }

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
   * Return the probability that the given observation came from the given
   * Gaussian component in this distribution.
   *
   * @param observation Observation to evaluate the probability of.
   * @param component Index of the component of the GMM to be considered.
   */
  double Probability(const arma::vec& observation,
                     const size_t component) const;

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this GMM.
   */
  arma::vec Random() const;

  /**
   * Estimate the probability distribution directly from the given observations,
   * using the given algorithm in the FittingType class to fit the data.
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
   * @tparam FittingType The type of fitting method which should be used
   *     (EMFit<> is suggested).
   * @param observations Observations of the model.
   * @param trials Number of trials to perform; the model in these trials with
   *      the greatest log-likelihood will be selected.
   * @param useExistingModel If true, the existing model is used as an initial
   *      model for the estimation.
   * @return The log-likelihood of the best fit.
   */
  template<typename FittingType = EMFit<>>
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
   * @return The log-likelihood of the best fit.
   */
  template<typename FittingType = EMFit<>>
  double Train(const arma::mat& observations,
               const arma::vec& probabilities,
               const size_t trials = 1,
               const bool useExistingModel = false,
               FittingType fitter = FittingType());

  /**
   * Classify the given observations as being from an individual component in
   * this GMM.  The resultant classifications are stored in the 'labels' object,
   * and each label will be between 0 and (Gaussians() - 1).  Supposing that a
   * point was classified with label 2, and that our GMM object was called
   * 'gmm', one could access the relevant Gaussian distribution as follows:
   *
   * @code
   * arma::vec mean = gmm.Means()[2];
   * arma::mat covariance = gmm.Covariances()[2];
   * double priorWeight = gmm.Weights()[2];
   * @endcode
   *
   * @param observations List of observations to classify.
   * @param labels Object which will be filled with labels.
   */
  void Classify(const arma::mat& observations,
                arma::Row<size_t>& labels) const;

  /**
   * Serialize the GMM.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * This function computes the loglikelihood of the given model.  This function
   * is used by GMM::Train().
   *
   * @param dataPoints Observations to calculate the likelihood for.
   * @param means Means of the given mixture model.
   * @param covars Covariances of the given mixture model.
   * @param weights Weights of the given mixture model.
   */
  double LogLikelihood(
      const arma::mat& dataPoints,
      const std::vector<distribution::GaussianDistribution>& distsL,
      const arma::vec& weights) const;
};

} // namespace gmm
} // namespace mlpack

// Include implementation.
#include "gmm_impl.hpp"

#endif

