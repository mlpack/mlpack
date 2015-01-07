/**
 * @file allow_empty_clusters.hpp
 * @author Ryan Curtin
 *
 * This very simple policy is used when K-Means is allowed to return empty
 * clusters.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_KMEANS_ALLOW_EMPTY_CLUSTERS_HPP
#define __MLPACK_METHODS_KMEANS_ALLOW_EMPTY_CLUSTERS_HPP

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
   * This function does nothing.  It is called by K-Means when K-Means detects
   * an empty cluster.
   *
   * @tparam MatType Type of data (arma::mat or arma::spmat).
   * @param data Dataset on which clustering is being performed.
   * @param emptyCluster Index of cluster which is empty.
   * @param centroids Centroids of each cluster (one per column).
   * @param clusterCounts Number of points in each cluster.
   * @param assignments Cluster assignments of each point.
   *
   * @return Number of points changed (0).
   */
  template<typename MatType>
  static size_t EmptyCluster(const MatType& /* data */,
                             const size_t /* emptyCluster */,
                             const MatType& /* centroids */,
                             arma::Col<size_t>& /* clusterCounts */,
                             arma::Col<size_t>& /* assignments */)
  {
    // Empty clusters are okay!  Do nothing.
    return 0;
  }
};

}; // namespace kmeans
}; // namespace mlpack

#endif
