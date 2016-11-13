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
#ifndef __MLPACK_METHODS_KMEANS_KILL_EMPTY_CLUSTERS_HPP
#define __MLPACK_METHODS_KMEANS_KILL_EMPTY_CLUSTERS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kmeans {

/**
 * Policy which allows K-Means to "kill" empty clusters without any error being
 * reported.  This means the centroids will be filled with DBL_MAX.
 */
class KillEmptyClusters
{
 public:
  //! Default constructor required by EmptyClusterPolicy policy.
  KillEmptyClusters() { }

  /**
   * This function sets an empty cluster found during k-means to all DBL_MAX
   * (i.e. an invalid "dead" cluster).
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
      const arma::mat& /* oldCentroids */,
      arma::mat& newCentroids,
      arma::Col<size_t>& /* clusterCounts */,
      MetricType& /* metric */,
      const size_t /* iteration */)
  {
    // Kill the empty cluster.
    newCentroids.col(emptyCluster).fill(DBL_MAX);
    return 0; // No points were changed.
  }

  //! Serialize the empty cluster policy (nothing to do).
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace kmeans
} // namespace mlpack

#endif
