/**
 * @file methods/kmeans/max_variance_new_cluster.hpp
 * @author Ryan Curtin
 *
 * An implementation of the EmptyClusterPolicy policy class for K-Means.  When
 * an empty cluster is detected, the point furthest from the centroid of the
 * cluster with maximum variance is taken to be a new cluster.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_HPP
#define MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

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
   * @tparam MatType Type of data (arma::mat or arma::spmat).
   * @param data Dataset on which clustering is being performed.
   * @param emptyCluster Index of cluster which is empty.
   * @param oldCentroids Centroids of each cluster (one per column) at the start
   *      of the iteration.
   * @param newCentroids Centroids of each cluster (one per column) at the end
   *      of the iteration.
   * @param clusterCounts Number of points in each cluster.
   * @param distance The distance metric to use.
   * @param iteration Number of iteration.
   *
   */
  template<typename DistanceType, typename MatType>
  void EmptyCluster(const MatType& data,
                    const size_t emptyCluster,
                    const arma::mat& oldCentroids,
                    arma::mat& newCentroids,
                    arma::Col<size_t>& clusterCounts,
                    DistanceType& distance,
                    const size_t iteration);

  //! Serialize the object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! Index of iteration for which variance is cached.
  size_t iteration;
  //! Cached variances for each cluster.
  arma::vec variances;
  //! Cached assignments for each point.
  arma::Row<size_t> assignments;

  //! Called when we are on a new iteration.
  template<typename DistanceType, typename MatType>
  void Precalculate(const MatType& data,
                    const arma::mat& oldCentroids,
                    arma::Col<size_t>& clusterCounts,
                    DistanceType& distance);
};

} // namespace mlpack

// Include implementation.
#include "max_variance_new_cluster_impl.hpp"

#endif
