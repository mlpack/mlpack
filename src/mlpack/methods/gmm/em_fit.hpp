/**
 * @file methods/gmm/em_fit.hpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Utility class to fit a GMM using the EM algorithm.  Used by
 * GMM::Estimate<>().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GMM_EM_FIT_HPP
#define MLPACK_METHODS_GMM_EM_FIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distributions/distributions.hpp>

// Default clustering mechanism.
#include <mlpack/methods/kmeans/kmeans.hpp>
// Default covariance matrix constraint.
#include "positive_definite_constraint.hpp"

namespace mlpack {

/**
 * This class contains methods which can fit a GMM to observations using the EM
 * algorithm.  It requires an initial clustering mechanism, which is by default
 * the KMeans algorithm.  The clustering mechanism must implement the following
 * method:
 *
 *  - void Cluster(const arma::mat& observations,
 *                 const size_t clusters,
 *                 arma::Row<size_t>& assignments);
 *
 * This method should create 'clusters' clusters, and return the assignment of
 * each point to a cluster.
 */
template<typename InitialClusteringType = KMeans<>,
         typename CovarianceConstraintPolicy = PositiveDefiniteConstraint,
         typename Distribution = GaussianDistribution<>>
class EMFit
{
 public:
  /**
   * Construct the EMFit object, optionally passing an InitialClusteringType
   * object (just in case it needs to store state).  Setting the maximum number
   * of iterations to 0 means that the EM algorithm will iterate until
   * convergence (with the given tolerance).
   *
   * The parameter forcePositive controls whether or not the covariance matrices
   * are checked for positive definiteness at each iteration.  This could be a
   * time-consuming task, so, if you know your data is well-behaved, you can set
   * it to false and save some runtime.
   *
   * @param maxIterations Maximum number of iterations for EM.
   * @param tolerance Log-likelihood tolerance required for convergence.
   * @param clusterer Object which will perform the initial clustering.
   * @param constraint Constraint policy of covariance.
   */
  EMFit(const size_t maxIterations = 300,
        const double tolerance = 1e-10,
        InitialClusteringType clusterer = InitialClusteringType(),
        CovarianceConstraintPolicy constraint = CovarianceConstraintPolicy());

  /**
   * Fit the observations to a Gaussian mixture model (GMM) using the EM
   * algorithm.  The size of the vectors (indicating the number of components)
   * must already be set.  Optionally, if useInitialModel is set to true, then
   * the model given in the means, covariances, and weights parameters is used
   * as the initial model, instead of using the InitialClusteringType::Cluster()
   * option.
   *
   * @param observations List of observations to train on.
   * @param dists Distributions to store model in.
   * @param weights Vector to store a priori weights in.
   * @param useInitialModel If true, the given model is used for the initial
   *      clustering.
   */
  void Estimate(const arma::mat& observations,
                std::vector<Distribution>& dists,
                arma::vec& weights,
                const bool useInitialModel = false);

  /**
   * Fit the observations to a Gaussian mixture model (GMM) using the EM
   * algorithm, taking into account the probabilities of each point being from
   * this mixture.  The size of the vectors (indicating the number of
   * components) must already be set.  Optionally, if useInitialModel is set to
   * true, then the model given in the means, covariances, and weights
   * parameters is used as the initial model, instead of using the
   * InitialClusteringType::Cluster() option.
   *
   * @param observations List of observations to train on.
   * @param probabilities Probability of each point being from this model.
   * @param dists Distributions to store model in.
   * @param weights Vector to store a priori weights in.
   * @param useInitialModel If true, the given model is used for the initial
   *      clustering.
   */
  void Estimate(const arma::mat& observations,
                const arma::vec& probabilities,
                std::vector<Distribution>& dists,
                arma::vec& weights,
                const bool useInitialModel = false);

  //! Get the clusterer.
  const InitialClusteringType& Clusterer() const { return clusterer; }
  //! Modify the clusterer.
  InitialClusteringType& Clusterer() { return clusterer; }

  //! Get the covariance constraint policy class.
  const CovarianceConstraintPolicy& Constraint() const { return constraint; }
  //! Modify the covariance constraint policy class.
  CovarianceConstraintPolicy& Constraint() { return constraint; }

  //! Get the maximum number of iterations of the EM algorithm.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations of the EM algorithm.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for the convergence of the EM algorithm.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for the convergence of the EM algorithm.
  double& Tolerance() { return tolerance; }

  //! Serialize the fitter.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  /**
   * Run the clusterer, and then turn the cluster assignments into Gaussians.
   * This is a helper function for both overloads of Estimate().  The vectors
   * must be already set to the number of clusters.
   *
   * @param observations List of observations.
   * @param means Vector to store means in.
   * @param dists Distributions to store model in.
   * @param weights Vector to store a priori weights in.
   */
  void InitialClustering(
      const arma::mat& observations,
      std::vector<Distribution>& dists,
      arma::vec& weights);

  /**
   * Calculate the log-likelihood of a model.  Yes, this is reimplemented in the
   * GMM code.  Intuition suggests that the log-likelihood is not the best way
   * to determine if the EM algorithm has converged.
   *
   * @param data Data matrix.
   * @param means Vector of means.
   * @param dists Distributions to store model in.
   * @param weights Vector of a priori weights.
   */
  double LogLikelihood(
      const arma::mat& data,
      const std::vector<Distribution>& dists,
      const arma::vec& weights) const;

  /**
   * Use the Armadillo gmm_diag clusterer to train a GMM with diagonal
   * covariance.  If InitialClusteringType == KMeans<>, this will use
   * Armadillo's initialization also.
   *
   * @param observations Data to train on.
   * @param dists Distributions to store model in.
   * @param weights Prior weights.
   * @param useInitialModel If true, the existing model will be used.
   */
  void ArmadilloGMMWrapper(
      const arma::mat& observations,
      std::vector<Distribution>& dists,
      arma::vec& weights,
      const bool useInitialModel);

  //! Maximum iterations of EM algorithm.
  size_t maxIterations;
  //! Tolerance for convergence of EM.
  double tolerance;
  //! Object which will perform the clustering.
  InitialClusteringType clusterer;
  //! Object which applies constraints to the covariance matrix.
  CovarianceConstraintPolicy constraint;
};

} // namespace mlpack

// Include implementation.
#include "em_fit_impl.hpp"

#endif
