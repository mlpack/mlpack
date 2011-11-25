/**
 * @file kmeans.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * K-Means clustering.
 */
#ifndef __MLPACK_METHODS_KMEANS_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_KMEANS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kmeans {

/**
 * This class implements K-Means clustering.
 */
class KMeans
{
 public:
  /**
   * Create a K-Means object and (optionally) set the parameters which K-Means
   * will be run with.  This implementation allows a few strategies to improve
   * the performance of K-Means, including "overclustering" and disallowing
   * empty clusters.
   *
   * The overclustering factor controls how many clusters are
   * actually found; for instance, with an overclustering factor of 4, if
   * K-Means is run to find 3 clusters, it will actually find 12, then merge the
   * nearest clusters until only 3 are left.
   *
   * @param overclusteringFactor Factor controlling how many extra clusters are
   *     found and then merged to get the desired number of clusters.
   * @param allowEmptyClusters If false, then clustering will fail instead of
   *     returning an empty cluster.
   * @param maxIterations Maximum number of iterations allowed before giving up
   *     (0 is valid, but the algorithm may never terminate).
   */
  KMeans(const double overclusteringFactor = 4.0,
         const bool allowEmptyClusters = false,
         const size_t maxIterations = 1000);

  /**
   * Perform K-Means clustering on the data, returning a list of cluster
   * assignments.  Optionally, the vector of assignments can be set to an
   * initial guess of the cluster assignments; to do this, the number of
   * elements in the list of assignments must be equal to the number of points
   * (columns) in the dataset.
   *
   * @param data Dataset to cluster.
   * @param clusters Number of clusters to compute.
   * @param assignments Vector to store cluster assignments in.  Can contain an
   *     initial guess at cluster assignments.
   */
  void Cluster(const arma::mat& data,
               const size_t clusters,
               arma::Col<size_t>& assignments) const;

  /**
   * Return the overclustering factor.
   */
  double OverclusteringFactor() const { return overclusteringFactor; }

  /**
   * Set the overclustering factor.
   */
  void OverclusteringFactor(const double overclusteringFactor)
  {
    if (overclusteringFactor < 1.0)
    {
      Log::Warn << "KMeans::OverclusteringFactor(): invalid value (<= 1.0) "
          "ignored." << std::endl;
      return;
    }

    this->overclusteringFactor = overclusteringFactor;
  }

  /**
   * Return whether or not empty clusters are allowed.
   */
  bool AllowEmptyClusters() const { return allowEmptyClusters; }

  /**
   * Set whether or not empty clusters are allowed.
   */
  void AllowEmptyClusters(bool allowEmptyClusters)
  {
    this->allowEmptyClusters = allowEmptyClusters;
  }

  /**
   * Get the maximum number of iterations.
   */
  size_t MaxIterations() const { return maxIterations; }

  /**
   * Set the maximum number of iterations.
   */
  void MaxIterations(size_t maxIterations)
  {
    this->maxIterations = maxIterations;
  }

 private:
  //! Factor controlling how many clusters are actually found.
  double overclusteringFactor;
  //! Whether or not to allow empty clusters to be returned.
  bool allowEmptyClusters;
  //! Maximum number of iterations before giving up.
  size_t maxIterations;
};

}; // namespace kmeans
}; // namespace mlpack

#endif // __MLPACK_METHODS_MOG_KMEANS_HPP
