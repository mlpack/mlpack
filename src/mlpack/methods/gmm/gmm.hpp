/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_em.h
 *
 * Defines a Gaussian Mixture model and
 * estimates the parameters of the model
 */
#ifndef __MLPACK_METHODS_MOG_MOG_EM_HPP
#define __MLPACK_METHODS_MOG_MOG_EM_HPP

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
 * from this GMM (see GMM::Estimate() for more information).
 *
 * The FittingType template class must provide a way for the GMM to train on
 * data.  It must provide the following two functions:
 *
 * @code
 * void Estimate(const arma::mat& observations,
 *               std::vector<arma::vec>& means,
 *               std::vector<arma::mat>& covariances,
 *               arma::vec& weights);
 *
 * void Estimate(const arma::mat& observations,
 *               const arma::vec& probabilities,
 *               std::vector<arma::vec>& means,
 *               std::vector<arma::mat>& covariances,
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
 * algorithm to train a GMM, and is the default fitting type.
 *
 * The GMM, once trained, can be used to generate random points from the
 * distribution and estimate the probability of points being from the
 * distribution.  The parameters of the GMM can be obtained through the
 * accessors and mutators.
 *
 * Example use:
 *
 * @code
 * // Set up a mixture of 5 gaussians in a 4-dimensional space (uses the default
 * // EM fitting mechanism).
 * GMM<> g(5, 4);
 *
 * // Train the GMM given the data observations.
 * g.Estimate(data);
 *
 * // Get the probability of 'observation' being observed from this GMM.
 * double probability = g.Probability(observation);
 *
 * // Get a random observation from the GMM.
 * arma::vec observation = g.Random();
 * @endcode
 */
template<typename FittingType = EMFit<> >
class GMM
{
 private:
  //! The number of Gaussians in the model.
  size_t gaussians;
  //! The dimensionality of the model.
  size_t dimensionality;
  //! Vector of means; one for each Gaussian.
  std::vector<arma::vec> means;
  //! Vector of covariances; one for each Gaussian.
  std::vector<arma::mat> covariances;
  //! Vector of a priori weights for each Gaussian.
  arma::vec weights;

 public:
  /**
   * Create an empty Gaussian Mixture Model, with zero gaussians.
   */
  GMM() :
      gaussians(0),
      dimensionality(0),
      localFitter(FittingType()),
      fitter(localFitter)
  {
    // Warn the user.  They probably don't want to do this.  If this constructor
    // is being used (because it is required by some template classes), the user
    // should know that it is potentially dangerous.
    Log::Debug << "GMM::GMM(): no parameters given; Estimate() may fail "
        << "unless parameters are set." << std::endl;
  }

  /**
   * Create a GMM with the given number of Gaussians, each of which have the
   * specified dimensionality.
   *
   * @param gaussians Number of Gaussians in this GMM.
   * @param dimensionality Dimensionality of each Gaussian.
   */
  GMM(const size_t gaussians, const size_t dimensionality) :
      gaussians(gaussians),
      dimensionality(dimensionality),
      means(gaussians, arma::vec(dimensionality)),
      covariances(gaussians, arma::mat(dimensionality, dimensionality)),
      weights(gaussians),
      localFitter(FittingType()),
      fitter(localFitter) { /* Nothing to do. */ }

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
  GMM(const size_t gaussians,
      const size_t dimensionality,
      FittingType& fitter) :
      gaussians(gaussians),
      dimensionality(dimensionality),
      means(gaussians, arma::vec(dimensionality)),
      covariances(gaussians, arma::mat(dimensionality, dimensionality)),
      weights(gaussians),
      fitter(fitter) { /* Nothing to do. */ }

  /**
   * Create a GMM with the given means, covariances, and weights.
   *
   * @param means Means of the model.
   * @param covariances Covariances of the model.
   * @param weights Weights of the model.
   */
  GMM(const std::vector<arma::vec>& means,
      const std::vector<arma::mat>& covariances,
      const arma::vec& weights) :
      gaussians(means.size()),
      dimensionality((!means.empty()) ? means[0].n_elem : 0),
      means(means),
      covariances(covariances),
      weights(weights),
      localFitter(FittingType()),
      fitter(localFitter) { /* Nothing to do. */ }

  /**
   * Create a GMM with the given means, covariances, and weights, and use the
   * given initialized FittingType class.  This is useful in cases where the
   * FittingType class needs to store some state.
   *
   * @param means Means of the model.
   * @param covariances Covariances of the model.
   * @param weights Weights of the model.
   */
  GMM(const std::vector<arma::vec>& means,
      const std::vector<arma::mat>& covariances,
      const arma::vec& weights,
      FittingType& fitter) :
      gaussians(means.size()),
      dimensionality((!means.empty()) ? means[0].n_elem : 0),
      means(means),
      covariances(covariances),
      weights(weights),
      fitter(fitter) { /* Nothing to do. */ }

  /**
   * Copy constructor for GMMs which use different fitting types.
   */
  template<typename OtherFittingType>
  GMM(const GMM<OtherFittingType>& other);

  /**
   * Copy constructor for GMMs using the same fitting type.  This also copies
   * the fitter.
   */
  GMM(const GMM& other);

  /**
   * Copy operator for GMMs which use different fitting types.
   */
  template<typename OtherFittingType>
  GMM& operator=(const GMM<OtherFittingType>& other);

  /**
   * Copy operator for GMMs which use the same fitting type.  This also copies
   * the fitter.
   */
  GMM& operator=(const GMM& other);

  //! Return the number of gaussians in the model.
  size_t Gaussians() const { return gaussians; }
  //! Modify the number of gaussians in the model.  Careful!  You will have to
  //! resize the means, covariances, and weights yourself.
  size_t& Gaussians() { return gaussians; }

  //! Return the dimensionality of the model.
  size_t Dimensionality() const { return dimensionality; }
  //! Modify the dimensionality of the model.  Careful!  You will have to update
  //! each mean and covariance matrix yourself.
  size_t& Dimensionality() { return dimensionality; }

  //! Return a const reference to the vector of means (mu).
  const std::vector<arma::vec>& Means() const { return means; }
  //! Return a reference to the vector of means (mu).
  std::vector<arma::vec>& Means() { return means; }

  //! Return a const reference to the vector of covariance matrices (sigma).
  const std::vector<arma::mat>& Covariances() const { return covariances; }
  //! Return a reference to the vector of covariance matrices (sigma).
  std::vector<arma::mat>& Covariances() { return covariances; }

  //! Return a const reference to the a priori weights of each Gaussian.
  const arma::vec& Weights() const { return weights; }
  //! Return a reference to the a priori weights of each Gaussian.
  arma::vec& Weights() { return weights; }

  //! Return a const reference to the fitting type.
  const FittingType& Fitter() const { return fitter; }
  //! Return a reference to the fitting type.
  FittingType& Fitter() { return fitter; }

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
   * @tparam FittingType The type of fitting method which should be used
   *     (EMFit<> is suggested).
   * @param observations Observations of the model.
   * @param trials Number of trials to perform; the model in these trials with
   *      the greatest log-likelihood will be selected.
   * @return The log-likelihood of the best fit.
   */
  double Estimate(const arma::mat& observations,
                  const size_t trials = 1);

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
   * @param observations Observations of the model.
   * @param probabilities Probability of each observation being from this
   *     distribution.
   * @param trials Number of trials to perform; the model in these trials with
   *     the greatest log-likelihood will be selected.
   * @return The log-likelihood of the best fit.
   */
  double Estimate(const arma::mat& observations,
                  const arma::vec& probabilities,
                  const size_t trials = 1);

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
                arma::Col<size_t>& labels) const;

 private:
  /**
   * This function computes the loglikelihood of the given model.  This function
   * is used by GMM::Estimate().
   *
   * @param dataPoints Observations to calculate the likelihood for.
   * @param means Means of the given mixture model.
   * @param covars Covariances of the given mixture model.
   * @param weights Weights of the given mixture model.
   */
  double LogLikelihood(const arma::mat& dataPoints,
                       const std::vector<arma::vec>& means,
                       const std::vector<arma::mat>& covars,
                       const arma::vec& weights) const;

  //! Locally-stored fitting object; in case the user did not pass one.
  FittingType localFitter;

  //! Reference to the fitting object we should use.
  FittingType& fitter;
};

}; // namespace gmm
}; // namespace mlpack

// Include implementation.
#include "gmm_impl.hpp"

#endif
