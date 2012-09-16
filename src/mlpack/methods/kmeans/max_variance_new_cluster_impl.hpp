/**
 * @file max_variance_new_cluster_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of MaxVarianceNewCluster class.
 *
 * This file is part of MLPACK 1.0.3.
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
        var(data.col(i) - centroids.col(assignments[i])));
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
          var(data.col(i) - centroids.col(maxVarCluster)));

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
