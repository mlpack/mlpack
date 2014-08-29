/**
 * @file em_fit.hpp
 * @author Ryan Curtin
 *
 * Utility class to fit a GMM using the EM algorithm.  Used by
 * GMM::Estimate<>().
 *
 * This file is part of MLPACK 1.0.10.
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
#ifndef __MLPACK_METHODS_GMM_EM_FIT_HPP
#define __MLPACK_METHODS_GMM_EM_FIT_HPP

#include <mlpack/core.hpp>

// Default clustering mechanism.
#include <mlpack/methods/kmeans/kmeans.hpp>
// Default covariance matrix constraint.
#include "positive_definite_constraint.hpp"

namespace mlpack {
namespace gmm {

/**
 * This class contains methods which can fit a GMM to observations using the EM
 * algorithm.  It requires an initial clustering mechanism, which is by default
 * the KMeans algorithm.  The clustering mechanism must implement the following
 * method:
 *
 *  - void Cluster(const arma::mat& observations,
 *                 const size_t clusters,
 *                 arma::Col<size_t>& assignments);
 *
 * This method should create 'clusters' clusters, and return the assignment of
 * each point to a cluster.
 */
template<typename InitialClusteringType = kmeans::KMeans<>,
         typename CovarianceConstraintPolicy = PositiveDefiniteConstraint>
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
   * @param forcePositive Check for positive-definiteness of each covariance
   *     matrix at each iteration.
   * @param clusterer Object which will perform the initial clustering.
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
   * @param means Vector to store trained means in.
   * @param covariances Vector to store trained covariances in.
   * @param weights Vector to store a priori weights in.
   * @param useInitialModel If true, the given model is used for the initial
   *      clustering.
   */
  void Estimate(const arma::mat& observations,
                std::vector<arma::vec>& means,
                std::vector<arma::mat>& covariances,
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
   * @param means Vector to store trained means in.
   * @param covariances Vector to store trained covariances in.
   * @param weights Vector to store a priori weights in.
   * @param useInitialModel If true, the given model is used for the initial
   *      clustering.
   */
  void Estimate(const arma::mat& observations,
                const arma::vec& probabilities,
                std::vector<arma::vec>& means,
                std::vector<arma::mat>& covariances,
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

 private:
  /**
   * Run the clusterer, and then turn the cluster assignments into Gaussians.
   * This is a helper function for both overloads of Estimate().  The vectors
   * must be already set to the number of clusters.
   *
   * @param observations List of observations.
   * @param means Vector to store means in.
   * @param covariances Vector to store covariances in.
   * @param weights Vector to store a priori weights in.
   */
  void InitialClustering(const arma::mat& observations,
                         std::vector<arma::vec>& means,
                         std::vector<arma::mat>& covariances,
                         arma::vec& weights);

  /**
   * Calculate the log-likelihood of a model.  Yes, this is reimplemented in the
   * GMM code.  Intuition suggests that the log-likelihood is not the best way
   * to determine if the EM algorithm has converged.
   *
   * @param data Data matrix.
   * @param means Vector of means.
   * @param covariances Vector of covariance matrices.
   * @param weights Vector of a priori weights.
   */
  double LogLikelihood(const arma::mat& data,
                       const std::vector<arma::vec>& means,
                       const std::vector<arma::mat>& covariances,
                       const arma::vec& weights) const;

  //! Maximum iterations of EM algorithm.
  size_t maxIterations;
  //! Tolerance for convergence of EM.
  double tolerance;
  //! Object which will perform the clustering.
  InitialClusteringType clusterer;
  //! Object which applies constraints to the covariance matrix.
  CovarianceConstraintPolicy constraint;
};

}; // namespace gmm
}; // namespace mlpack

// Include implementation.
#include "em_fit_impl.hpp"

#endif
