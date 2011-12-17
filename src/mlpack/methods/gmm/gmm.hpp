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

namespace mlpack {
namespace gmm /** Gaussian Mixture Models. */ {

/**
 * A Gaussian Mixture Model (GMM). This class uses maximum likelihood loss
 * functions to estimate the parameters of the GMM on a given dataset via the EM
 * algorithm.  The GMM can be trained either with labeled or unlabeled data.
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
  GMM() : gaussians(0), dimensionality(0)
  {
    // Warn the user.  They probably don't want to do this.  If this constructor
    // is being used (because it is required by some template classes), the user
    // should know that it is potentially dangerous.
    Log::Debug << "GMM::GMM(): no parameters given; Estimate() will fail "
        << "unless parameters are set." << std::endl;
  }

  /**
   * Create a GMM with the given number of Gaussians, each of which have the
   * specified dimensionality.
   *
   * @param gaussians Number of Gaussians in this GMM.
   * @param dimensionality Dimensionality of each Gaussian.
   */
  GMM(size_t gaussians, size_t dimensionality) :
      gaussians(gaussians),
      dimensionality(dimensionality),
      means(gaussians, arma::vec(dimensionality)),
      covariances(gaussians, arma::mat(dimensionality, dimensionality)),
      weights(gaussians) { /* nothing to do */ }

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
      dimensionality((means.size() > 0) ? means[0].n_elem : 0),
      means(means),
      covariances(covariances),
      weights(weights) { /* nothing to do */ }

  //! Return the number of gaussians in the model.
  size_t Gaussians() const { return gaussians; }

  //! Return the dimensionality of the model.
  size_t Dimensionality() const { return dimensionality; }

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

  /**
   * Return the probability that the given observation came from this
   * distribution.
   *
   * @param observation Observation to evaluate the probability of.
   */
  double Probability(const arma::vec& observation) const;

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this GMM.
   */
  arma::vec Random() const;

  /**
   * Estimate the probability distribution directly from the given observations,
   * using the EM algorithm to obtain the Maximum Likelihood parameter.
   *
   * @param observations Observations of the model.
   */
  void Estimate(const arma::mat& observations);

  /**
   * Estimate the probability distribution directly from the given observations,
   * taking into account the probability of each observation actually being from
   * this distribution.
   *
   * @param observations Observations of the model.
   * @param probability Probability of each observation.
   */
  void Estimate(const arma::mat& observations,
                const arma::vec& probabilities);

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
  double Loglikelihood(const arma::mat& dataPoints,
                       const std::vector<arma::vec>& means,
                       const std::vector<arma::mat>& covars,
                       const arma::vec& weights) const;

  /**
   * This function uses the given clustering class and initializes means,
   * covariances, and weights into the passed objects based on the assignments
   * of the clustering class.
   *
   * @param clusterer Initialized clustering class (must implement void
   *      Cluster(const arma::mat&, arma::Col<size_t>&)
   * @param data Dataset to perform clustering on.
   * @param means Vector to store means in.
   * @param covars Vector to store covariances in.
   * @param weights Vector to store weights in.
   */
  template<typename ClusteringType>
  void InitialClustering(const ClusteringType& clusterer,
                         const arma::mat& data,
                         std::vector<arma::vec>& means,
                         std::vector<arma::mat>& covars,
                         arma::vec& weights) const;
};

}; // namespace gmm
}; // namespace mlpack

// Include implementation.
#include "gmm_impl.hpp"

#endif
