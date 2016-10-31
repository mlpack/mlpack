/**
 * @file allow_empty_clusters.hpp
 * @author Ryan Curtin
 *
 * This very simple policy is used when K-Means is allowed to return empty
 * clusters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_ALLOW_EMPTY_CLUSTERS_HPP
#define MLPACK_METHODS_KMEANS_ALLOW_EMPTY_CLUSTERS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kmeans {

/**
 * Policy which allows K-Means to create empty clusters without any error being
 * reported.
 */
class AllowEmptyClusters
{
 public:
  //! Default constructor required by EmptyClusterPolicy policy.
  AllowEmptyClusters() { }

  /**
   * This function allows empty clusters to persist simply by leaving the empty
   * cluster in its last position.
   *
   * @tparam MatType Type of data (arma::mat or arma::spmat).
   * @param data Dataset on which clustering is being performed.
   * @param emptyCluster Index of cluster which is empty.
   * @param oldCentroids Centroids of each cluster (one per column) at the start
   *      of the iteration.
   * @param newCentroids Centroids of each cluster (one per column) at the end
   *      of the iteration.
   * @param clusterCounts Number of points in each cluster.
   * @param assignments Cluster assignments of each point.
   * @param iteration Number of iteration.
   *
   * @return Number of points changed (0).
   */
  template<typename MetricType, typename MatType>
  static inline force_inline size_t EmptyCluster(
      const MatType& /* data */,
      const size_t emptyCluster,
      const arma::mat& oldCentroids,
      arma::mat& newCentroids,
      arma::Col<size_t>& /* clusterCounts */,
      MetricType& /* metric */,
      const size_t /* iteration */)
  {
    // Take the last iteration's centroid.
    newCentroids.col(emptyCluster) = oldCentroids.col(emptyCluster);
    return 0; // No points were changed.
  }

  //! Serialize the empty cluster policy (nothing to do).
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace kmeans
} // namespace mlpack

#endif
