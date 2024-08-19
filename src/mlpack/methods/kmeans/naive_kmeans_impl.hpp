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

  // Determine the number of threads and calculate segment size
  size_t numThreads = 1;
  #ifdef MLPACK_USE_OPENMP
    numThreads = omp_get_max_threads();
  #endif

  const size_t points = dataset.n_cols;
  const size_t nominalSegmentSize = (points + numThreads - 1) / numThreads; // Ceiling division

  // Pre-allocate thread-local storage
  std::vector<arma::mat> threadCentroids(numThreads, arma::mat(centroids.n_rows, centroids.n_cols));
  std::vector<arma::Col<size_t>> threadCounts(numThreads, arma::Col<size_t>(centroids.n_cols));

  // Precompute squared norms of centroids
  arma::vec centroidNorms(centroids.n_cols);
  #pragma omp parallel for
  for (size_t j = 0; j < centroids.n_cols; ++j)
  {
    centroidNorms(j) = arma::dot(centroids.col(j), centroids.col(j));
  }

  #pragma omp parallel
  {
    size_t threadId = 0;
    #ifdef MLPACK_USE_OPENMP
      threadId = omp_get_thread_num();
    #endif

    arma::mat& localCentroids = threadCentroids[threadId];
    arma::Col<size_t>& localCounts = threadCounts[threadId];

    #pragma omp for
    for (size_t i = 0; i < points; ++i)
    {
      // Use a temporary dense vector for both sparse and dense matrices
      arma::vec dataPoint;
      if (std::is_same<MatType, arma::SpMat<typename MatType::elem_type>>::value)
        dataPoint = arma::vec(arma::conv_to<arma::vec>::from(dataset.col(i)));
      else
        dataPoint = dataset.col(i);

      double dataNorm = arma::dot(dataPoint, dataPoint);

      double minDistance = std::numeric_limits<double>::infinity();
      size_t closestCluster = centroids.n_cols; // Invalid value.

      for (size_t j = 0; j < centroids.n_cols; ++j)
      {
        const double dist = std::max(0.0, dataNorm + centroidNorms(j) - 2 * arma::dot(dataPoint, centroids.col(j)));
        if (dist < minDistance)
        {
          minDistance = dist;
          closestCluster = j;
        }
      }

      Log::Assert(closestCluster != centroids.n_cols);

      // We now have the minimum distance centroid index.  Update that centroid.
      localCentroids.unsafe_col(closestCluster) += dataPoint;
      localCounts(closestCluster)++;
    }

    // Combine calculated state from each thread
    #pragma omp critical
    {
      newCentroids += localCentroids;
      counts += localCounts;
    }
  }

  // Now normalize the centroid.
  for (size_t i = 0; i < centroids.n_cols; ++i)
    if (counts(i) != 0)
      newCentroids.col(i) /= counts(i);

  distanceCalculations += centroids.n_cols * dataset.n_cols;

  // Calculate cluster distortion for this iteration.
  double cNorm = 0.0;
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    cNorm += std::pow(distance.Evaluate(centroids.col(i), newCentroids.col(i)), 2.0);
  }
  distanceCalculations += centroids.n_cols;

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif