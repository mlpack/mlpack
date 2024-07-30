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

  // Determine the number of threads
  size_t numThreads = 1;
  #ifdef MLPACK_USE_OPENMP
    numThreads = static_cast<size_t>(omp_get_max_threads());
  #endif

  // Pre-allocate thread-local storage
  std::vector<arma::mat> threadCentroids(numThreads, arma::mat(dims, clusters));
  std::vector<arma::Col<size_t>> threadCounts(numThreads, arma::Col<size_t>(clusters));

  double cNorm = 0.0;

  #pragma omp parallel reduction(+:cNorm)
  {
    const size_t threadId = 
    #ifdef MLPACK_USE_OPENMP
      omp_get_thread_num();
    #else
      0;
    #endif

    arma::mat& localCentroids = threadCentroids[threadId];
    arma::Col<size_t>& localCounts = threadCounts[threadId];

    #pragma omp for schedule(static)
    for (size_t i = 0; i < points; ++i)
    {
      size_t closestCluster = 0;
      double minDistance = std::numeric_limits<double>::max();
      const arma::vec& point = dataset.col(i);

      for (size_t j = 0; j < clusters; ++j)
      {
        const double dist = distance.Evaluate(point, centroids.col(j));
        if (dist < minDistance)
        {
          minDistance = dist;
          closestCluster = j;
        }
      }

      localCentroids.col(closestCluster) += point;
      localCounts[closestCluster]++;
    }
  }

  // Combine results
  for (size_t t = 0; t < numThreads; ++t)
  {
    newCentroids += threadCentroids[t];
    counts += threadCounts[t];
  }

  // Normalize the centroids and calculate distortion
  #pragma omp parallel for reduction(+:cNorm) schedule(static)
  for (size_t j = 0; j < clusters; ++j)
  {
    if (counts[j] > 0)
    {
      newCentroids.col(j) /= counts[j];
      cNorm += std::pow(distance.Evaluate(centroids.col(j), newCentroids.col(j)), 2.0);
    }
    else
    {
      newCentroids.col(j) = centroids.col(j);
    }
  }

  distanceCalculations += clusters * points;

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif