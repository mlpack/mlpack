/**
 * @file max_variance_new_cluster.hpp
 * @author Ryan Curtin
 *
 * An implementation of the EmptyClusterPolicy policy class for K-Means.  When
 * an empty cluster is detected, the point furthest from the centroid of the
 * cluster with maximum variance is taken to be a new cluster.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
  MaxVarianceNewCluster() { }

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
  template<typename MatType>
  static size_t EmptyCluster(const MatType& data,
                             const size_t emptyCluster,
                             const MatType& centroids,
                             arma::Col<size_t>& clusterCounts,
                             arma::Col<size_t>& assignments);
};

}; // namespace kmeans
}; // namespace mlpack

// Include implementation.
#include "max_variance_new_cluster_impl.hpp"

#endif
