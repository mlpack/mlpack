/**
 * @file methods/kmeans/naive_kmeans_impl.hpp
 * @author Ryan Curtin
 * @author Shikhar Bhardwaj
 *
 * An implementation of a naively-implemented step of the Lloyd algorithm for
 * k-means clustering, using OpenMP for parallelization over multiple threads.
 * This may still be the best choice for small datasets or datasets with very
 * high dimensionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_NAIVE_KMEANS_IMPL_HPP
#define MLPACK_METHODS_KMEANS_NAIVE_KMEANS_IMPL_HPP

// In case it hasn't been included yet.
#include "naive_kmeans.hpp"

namespace mlpack {

template<typename DistanceType, typename MatType>
NaiveKMeans<DistanceType, MatType>::NaiveKMeans(const MatType& dataset,
                                                DistanceType& distance) :
    dataset(dataset),
    distance(distance),
    distanceCalculations(0)
{ /* Nothing to do. */ }

// Run a single iteration.
template<typename DistanceType, typename MatType>
double NaiveKMeans<DistanceType, MatType>::Iterate(const arma::mat& centroids,
                                                   arma::mat& newCentroids,
                                                   arma::Col<size_t>& counts)
{
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  const size_t dimensions = centroids.n_rows;
  const size_t numClusters = centroids.n_cols;
  const size_t numPoints = dataset.n_cols;

  #pragma omp parallel
  {
    // Thread-local storage for partial sums
    arma::mat threadCentroids(dimensions, numClusters, arma::fill::zeros);
    arma::Col<size_t> threadCounts(numClusters, arma::fill::zeros);

    #pragma omp for schedule(static) nowait
    for (size_t i = 0; i < numPoints; ++i)
    {
      size_t closestCluster = 0;
      double minDistance = std::numeric_limits<double>::max();

      // Find the closest centroid
      for (size_t j = 0; j < numClusters; ++j)
      {
        double dist = distance.Evaluate(dataset.col(i), centroids.col(j));
        if (dist < minDistance)
        {
          minDistance = dist;
          closestCluster = j;
        }
      }

      // Update thread-local centroids and counts
      threadCentroids.col(closestCluster) += dataset.col(i);
      threadCounts(closestCluster)++;
    }

    // Reduce thread-local results to shared variables
    #pragma omp critical
    {
      newCentroids += threadCentroids;
      counts += threadCounts;
    }
  }

  // Normalize centroids
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < numClusters; ++i)
  {
    if (counts(i) > 0)
      newCentroids.col(i) /= counts(i);
  }

  distanceCalculations += numClusters * numPoints;

  // Calculate cluster distortion
  double cNorm = 0.0;
  #pragma omp parallel for reduction(+:cNorm) schedule(static)
  for (size_t i = 0; i < numClusters; ++i)
  {
    cNorm += std::pow(distance.Evaluate(centroids.col(i), newCentroids.col(i)), 2.0);
  }
  distanceCalculations += numClusters;

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif