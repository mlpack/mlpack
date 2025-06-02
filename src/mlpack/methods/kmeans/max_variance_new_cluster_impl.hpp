/**
 * @file methods/kmeans/max_variance_new_cluster_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of MaxVarianceNewCluster class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_IMPL_HPP
#define MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_IMPL_HPP

// Just in case it has not been included.
#include "max_variance_new_cluster.hpp"

namespace mlpack {

/**
 * Take action about an empty cluster.
 */
template<typename DistanceType, typename MatType>
void MaxVarianceNewCluster::EmptyCluster(const MatType& data,
                                         const size_t emptyCluster,
                                         const arma::mat& oldCentroids,
                                         arma::mat& newCentroids,
                                         arma::Col<size_t>& clusterCounts,
                                         DistanceType& distance,
                                         const size_t iteration)
{
  // If necessary, calculate the variances and assignments.
  if (iteration != this->iteration || assignments.n_elem != data.n_cols)
    Precalculate(data, oldCentroids, clusterCounts, distance);
  this->iteration = iteration;

  // Now find the cluster with maximum variance.
  arma::uword maxVarCluster = variances.index_max();

  // If the cluster with maximum variance has variance of 0, then we can't
  // continue.  All the points are the same.
  if (variances[maxVarCluster] == 0.0)
    return;

  // Now, inside this cluster, find the point which is furthest away.
  size_t furthestPoint = data.n_cols;
  double maxDistance = -DBL_MAX;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (assignments[i] == maxVarCluster)
    {
      const double dist = std::pow(distance.Evaluate(data.col(i),
          newCentroids.col(maxVarCluster)), 2.0);

      if (dist > maxDistance)
      {
        maxDistance = dist;
        furthestPoint = i;
      }
    }
  }

  // Take that point and add it to the empty cluster.
  newCentroids.col(maxVarCluster) *= (double(clusterCounts[maxVarCluster]) /
      double(clusterCounts[maxVarCluster] - 1));
  newCentroids.col(maxVarCluster) -= (1.0 / (clusterCounts[maxVarCluster] -
      1.0)) * arma::vec(data.col(furthestPoint));
  clusterCounts[maxVarCluster]--;
  clusterCounts[emptyCluster]++;
  newCentroids.col(emptyCluster) = arma::vec(data.col(furthestPoint));
  assignments[furthestPoint] = emptyCluster;

  // Modify the variances, as necessary.
  variances[emptyCluster] = 0;
  // One has already been subtracted from clusterCounts[maxVarCluster].  If
  // EmptyCluster() is called again, we can't pull another point from
  // maxVarCluster (we'd be making an empty cluster), so force a call to
  // Precalculate() if EmptyCluster() is called again by changing
  // this->iteration.
  if (clusterCounts[maxVarCluster] <= 1)
  {
    variances[maxVarCluster] = 0;
    --this->iteration;
  }
  else
  {
    variances[maxVarCluster] = (1.0 / clusterCounts[maxVarCluster]) *
        ((clusterCounts[maxVarCluster] + 1) * variances[maxVarCluster] -
        maxDistance);
  }

  // Output some debugging information.
  Log::Debug << "Point " << furthestPoint << " assigned to empty cluster " <<
      emptyCluster << ".\n";
}

//! Serialize the object.
template<typename Archive>
void MaxVarianceNewCluster::serialize(Archive& /* ar */,
                                      const uint32_t /* version */)
{
  // Serialization is useless here, because the only thing we store is
  // precalculated quantities, and if we're serializing, our precalculations are
  // likely to be useless when we deserialize (because the user will be running
  // a different clustering, probably).  So there is no need to store anything,
  // and if we are loading, we just reset the assignments array so
  // precalculation will happen next time EmptyCluster() is called.
  if (cereal::is_loading<Archive>())
    assignments.set_size(0);
}

template<typename DistanceType, typename MatType>
void MaxVarianceNewCluster::Precalculate(const MatType& data,
                                         const arma::mat& oldCentroids,
                                         arma::Col<size_t>& clusterCounts,
                                         DistanceType& distance)
{
  // We have to calculate the variances of each cluster and the assignments of
  // each point.  This is most easily done by iterating through the entire
  // dataset.
  variances.zeros(oldCentroids.n_cols);
  assignments.set_size(data.n_cols);

  // Add the variance of each point's distance away from the cluster.  I think
  // this is the sensible thing to do.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Find the closest centroid to this point.
    double minDistance = std::numeric_limits<double>::infinity();
    size_t closestCluster = oldCentroids.n_cols; // Invalid value.

    for (size_t j = 0; j < oldCentroids.n_cols; ++j)
    {
      const double dist = distance.Evaluate(data.col(i), oldCentroids.col(j));

      if (dist < minDistance)
      {
        minDistance = dist;
        closestCluster = j;
      }
    }

    assignments[i] = closestCluster;
    variances[closestCluster] += std::pow(distance.Evaluate(data.col(i),
        oldCentroids.col(closestCluster)), 2.0);
  }

  // Divide by the number of points in the cluster to produce the variance,
  // unless the cluster is empty or contains only one point, in which case we
  // set the variance to 0.
  for (size_t i = 0; i < clusterCounts.n_elem; ++i)
    if (clusterCounts[i] <= 1)
      variances[i] = 0;
    else
      variances[i] /= clusterCounts[i];
}

} // namespace mlpack

#endif
