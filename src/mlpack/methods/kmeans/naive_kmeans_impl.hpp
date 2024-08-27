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

#include "naive_kmeans.hpp"

namespace mlpack {

template<typename DistanceType, typename MatType>
NaiveKMeans<DistanceType, MatType>::NaiveKMeans(const MatType& dataset,
                                                DistanceType& distance) :
    dataset(dataset),
    distance(distance),
    distanceCalculations(0)
{ /* Nothing to do. */ }

template<typename DistanceType, typename MatType>
double NaiveKMeans<DistanceType, MatType>::Iterate(const arma::mat& centroids,
                                                   arma::mat& newCentroids,
                                                   arma::Col<size_t>& counts)
{
  const size_t dims = centroids.n_rows;
  const size_t numCentroids = centroids.n_cols;
  const size_t numPoints = dataset.n_cols;

  newCentroids.zeros(dims, numCentroids);
  counts.zeros(numCentroids);

  // Pre-allocate thread-local storage
  const int numThreads = omp_get_max_threads();
  std::vector<arma::mat> threadCentroids(numThreads, arma::mat(dims, numCentroids, arma::fill::zeros));
  std::vector<arma::Col<size_t>> threadCounts(numThreads, arma::Col<size_t>(numCentroids, arma::fill::zeros));

  #pragma omp parallel
  {
    const int threadId = omp_get_thread_num();
    auto& localCentroids = threadCentroids[threadId];
    auto& localCounts = threadCounts[threadId];

    #pragma omp for schedule(static)
    for (size_t i = 0; i < numPoints; ++i)
    {
      double minDistance = std::numeric_limits<double>::max();
      size_t closestCluster = numCentroids;

      const auto& point = dataset.col(i);

      for (size_t j = 0; j < numCentroids; ++j)
      {
        const double dist = distance.Evaluate(point, centroids.col(j));
        if (dist < minDistance)
        {
          minDistance = dist;
          closestCluster = j;
        }
      }

      localCentroids.col(closestCluster) += point;
      localCounts(closestCluster)++;
    }
  }

  // Combine results from all threads
  for (int t = 0; t < numThreads; ++t)
  {
    newCentroids += threadCentroids[t];
    counts += threadCounts[t];
  }

  // Normalize the centroids
  for (size_t i = 0; i < numCentroids; ++i)
  {
    if (counts(i) != 0)
      newCentroids.col(i) /= counts(i);
  }

  distanceCalculations += numCentroids * numPoints;

  // Calculate cluster distortion
  double cNorm = 0.0;
  #pragma omp parallel for reduction(+:cNorm) schedule(static)
  for (size_t i = 0; i < numCentroids; ++i)
  {
    cNorm += std::pow(distance.Evaluate(centroids.col(i), newCentroids.col(i)), 2.0);
  }
  distanceCalculations += numCentroids;

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif