/**
 * @file max_variance_new_cluster_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of MaxVarianceNewCluster class.
 */
#ifndef __MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_IMPL_HPP

// Just in case it has not been included.
#include "max_variance_new_cluster.hpp"

namespace mlpack {
namespace kmeans {

/**
 * Take action about an empty cluster.
 */
template<typename MatType>
size_t MaxVarianceNewCluster::EmptyCluster(const MatType& data,
                                           const size_t emptyCluster,
                                           const MatType& centroids,
                                           arma::Col<size_t>& clusterCounts,
                                           arma::Col<size_t>& assignments)
{
  // First, we need to find the cluster with maximum variance (by which I mean
  // the sum of the covariance matrices).
  arma::vec variances;
  variances.zeros(clusterCounts.n_elem); // Start with 0.

  // Add the variance of each point's distance away from the cluster.  I think
  // this is the sensible thing to do.
  for (size_t i = 0; i < data.n_cols; i++)
  {
    variances[assignments[i]] += arma::as_scalar(
        arma::var(data.col(i) - centroids.col(assignments[i])));
  }

  // Now find the cluster with maximum variance.
  arma::uword maxVarCluster;
  variances.max(maxVarCluster);

  // Now, inside this cluster, find the point which is furthest away.
  size_t furthestPoint = data.n_cols;
  double maxDistance = 0;
  for (size_t i = 0; i < data.n_cols; i++)
  {
    if (assignments[i] == maxVarCluster)
    {
      double distance = arma::as_scalar(
          arma::var(data.col(i) - centroids.col(maxVarCluster)));

      if (distance > maxDistance)
      {
        maxDistance = distance;
        furthestPoint = i;
      }
    }
  }

  // Take that point and add it to the empty cluster.
  clusterCounts[maxVarCluster]--;
  clusterCounts[emptyCluster]++;
  assignments[furthestPoint] = emptyCluster;

  // Output some debugging information.
  Log::Debug << "Point " << furthestPoint << " assigned to empty cluster " <<
      emptyCluster << ".\n";

  return 1; // We only changed one point.
}

}; // namespace kmeans
}; // namespace mlpack

#endif
