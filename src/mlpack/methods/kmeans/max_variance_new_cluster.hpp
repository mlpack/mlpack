/**
 * @file max_variance_new_cluster.hpp
 * @author Ryan Curtin
 *
 * An implementation of the EmptyClusterPolicy policy class for K-Means.  When
 * an empty cluster is detected, the point furthest from the centroid of the
 * cluster with maximum variance is taken to be a new cluster.
 */
#ifndef __MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_HPP
#define __MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kmeans {

/**
 * When an empty cluster is detected, this class takes the point furthest from
 * the centroid of the cluster with maximum variance as a new cluster.
 */
class MaxVarianceNewCluster
{
 public:
  //! Default constructor required by EmptyClusterPolicy.
  MaxVarianceNewCluster() : iteration(size_t(-1)) { }

  /**
   * Take the point furthest from the centroid of the cluster with maximum
   * variance to be a new cluster.
   *
   * @tparam MatType Type of data (arma::mat or arma::sp_mat).
   * @param data Dataset on which clustering is being performed.
   * @param emptyCluster Index of cluster which is empty.
   * @param centroids Centroids of each cluster (one per column).
   * @param clusterCounts Number of points in each cluster.
   * @param assignments Cluster assignments of each point.
   *
   * @return Number of points changed.
   */
  template<typename MetricType, typename MatType>
  size_t EmptyCluster(const MatType& data,
                      const size_t emptyCluster,
                      arma::mat& centroids,
                      arma::Col<size_t>& clusterCounts,
                      MetricType& metric,
                      const size_t iteration);

 private:
  //! Index of iteration for which variance is cached.
  size_t iteration;
  //! Cached variances for each cluster.
  arma::vec variances;
  //! Cached assignments for each point.
  arma::Col<size_t> assignments;

  //! Called when we are on a new iteration.
  template<typename MetricType, typename MatType>
  void Precalculate(const MatType& data,
                    arma::mat& centroids,
                    arma::Col<size_t>& clusterCounts,
                    MetricType& metric);
};

} // namespace kmeans
} // namespace mlpack

// Include implementation.
#include "max_variance_new_cluster_impl.hpp"

#endif
