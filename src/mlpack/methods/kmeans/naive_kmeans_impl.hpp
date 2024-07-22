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

#ifdef MLPACK_USE_OPENMP
  #include <omp.h>
#endif

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
  const size_t dims = dataset.n_rows;
  const size_t points = dataset.n_cols;
  const size_t clusters = centroids.n_cols;

  newCentroids.zeros(dims, clusters);
  counts.zeros(clusters);

  // Pre-compute squared norms of centroids
  arma::vec centroidNorms(clusters);
  #ifdef MLPACK_USE_OPENMP
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t j = 0; j < clusters; ++j)
  {
    centroidNorms(j) = arma::dot(centroids.col(j), centroids.col(j));
  }

  // Determine the number of threads and calculate segment size
  size_t effectiveThreads = 1;
  #ifdef MLPACK_USE_OPENMP
    const size_t numThreads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
    const size_t minVectorsPerThread = 100; 
    effectiveThreads = std::min(numThreads, points / minVectorsPerThread);
  #endif
  const size_t nominalSegmentSize = points / effectiveThreads;

  // Pre-allocate thread-local storage
  std::vector<arma::mat> threadCentroids(effectiveThreads, arma::mat(dims, clusters, arma::fill::zeros));
  std::vector<arma::Col<size_t>> threadCounts(effectiveThreads, arma::Col<size_t>(clusters, arma::fill::zeros));

  #ifdef MLPACK_USE_OPENMP
    #pragma omp parallel num_threads(effectiveThreads)
  #endif
  {
    size_t threadId = 0;
    #ifdef MLPACK_USE_OPENMP
      threadId = omp_get_thread_num();
    #endif
    const size_t segmentStart = threadId * nominalSegmentSize;
    const size_t segmentEnd = (threadId == effectiveThreads - 1) ? points : (threadId + 1) * nominalSegmentSize;

    arma::mat& localCentroids = threadCentroids[threadId];
    arma::Col<size_t>& localCounts = threadCounts[threadId];

    arma::vec distances(clusters);

    for (size_t i = segmentStart; i < segmentEnd; ++i)
    {
      const auto dataPoint = dataset.col(i);
      const double dataNorm = arma::dot(dataPoint, dataPoint);

      // Calculate distances to all centroids
      for (size_t j = 0; j < clusters; ++j)
      {
        const arma::vec& centroid = centroids.col(j);
        distances(j) = dataNorm + centroidNorms(j) - 2 * arma::dot(dataPoint, centroid);
      }

      // Find the closest centroid
      const size_t closestCluster = distances.index_min();

      // Update local centroids and counts
      localCentroids.col(closestCluster) += dataPoint;
      localCounts(closestCluster)++;
    }
  }

  // Combine results from all threads
  for (size_t t = 0; t < effectiveThreads; ++t)
  {
    newCentroids += threadCentroids[t];
    counts += threadCounts[t];
  }

  // Normalize the centroids
  for (size_t j = 0; j < clusters; ++j)
  {
    if (counts(j) > 0)
    {
      newCentroids.col(j) /= counts(j);
    }
  }

  // Calculate cluster distortion
  double cNorm = 0.0;
  #ifdef MLPACK_USE_OPENMP
    #pragma omp parallel for reduction(+:cNorm) schedule(static)
  #endif
  for (size_t j = 0; j < clusters; ++j)
  {
    cNorm += arma::norm(centroids.col(j) - newCentroids.col(j), 2);
  }

  distanceCalculations += clusters * points;

  return cNorm;
}

} // namespace mlpack

#endif