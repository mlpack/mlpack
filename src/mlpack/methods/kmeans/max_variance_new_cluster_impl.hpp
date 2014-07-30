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
template<typename MetricType, typename MatType>
size_t MaxVarianceNewCluster::EmptyCluster(const MatType& data,
                                           const size_t emptyCluster,
                                           arma::mat& centroids,
                                           arma::Col<size_t>& clusterCounts,
                                           MetricType& metric)
{
  // First, we need to find the cluster with maximum variance (by which I mean
  // the sum of the covariance matrices).
  arma::vec variances;
  variances.zeros(clusterCounts.n_elem); // Start with 0.
  arma::Col<size_t> assignments(data.n_cols);

  // Add the variance of each point's distance away from the cluster.  I think
  // this is the sensible thing to do.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Find the closest centroid to this point.
    double minDistance = std::numeric_limits<double>::infinity();
    size_t closestCluster = centroids.n_cols; // Invalid value.

    for (size_t j = 0; j < centroids.n_cols; j++)
    {
      const double distance = metric.Evaluate(data.col(i), centroids.col(j));

      if (distance < minDistance)
      {
        minDistance = distance;
        closestCluster = j;
      }
    }

    assignments[i] = closestCluster;
    variances[closestCluster] += metric.Evaluate(data.col(i),
        centroids.col(closestCluster));
  }

  // Divide by the number of points in the cluster to produce the variance.
  // Although a -nan will occur here for the empty cluster(s), this doesn't
  // matter because variances.max() won't pick it up.  If the number of points
  // in the cluster is 1, we ensure that cluster is not selected by forcing the
  // variance to 0.
  for (size_t i = 0; i < clusterCounts.n_elem; ++i)
    variances[i] /= (clusterCounts[i] == 1) ? DBL_MAX : clusterCounts[i];

  // Now find the cluster with maximum variance.
  arma::uword maxVarCluster;
  variances.max(maxVarCluster);

  // Now, inside this cluster, find the point which is furthest away.
  size_t furthestPoint = data.n_cols;
  double maxDistance = -DBL_MAX;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (assignments[i] == maxVarCluster)
    {
      const double distance = metric.Evaluate(data.col(i),
          centroids.col(maxVarCluster));

      if (distance > maxDistance)
      {
        maxDistance = distance;
        furthestPoint = i;
      }
    }
  }

  // Take that point and add it to the empty cluster.
  centroids.col(maxVarCluster) *= (double(clusterCounts[maxVarCluster]) /
      double(clusterCounts[maxVarCluster] - 1));
  centroids.col(maxVarCluster) -= (1.0 / (clusterCounts[maxVarCluster] - 1.0)) *
      arma::vec(data.col(furthestPoint));
  clusterCounts[maxVarCluster]--;
  clusterCounts[emptyCluster]++;
  centroids.col(emptyCluster) = arma::vec(data.col(furthestPoint));
  assignments[furthestPoint] = emptyCluster;

  // Output some debugging information.
  Log::Debug << "Point " << furthestPoint << " assigned to empty cluster " <<
      emptyCluster << ".\n";

  return 1; // We only changed one point.
}

}; // namespace kmeans
}; // namespace mlpack

#endif
