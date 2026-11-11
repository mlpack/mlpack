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

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename DistanceType, typename MatType, typename CentroidsType>
NaiveKMeans<DistanceType, MatType, CentroidsType>::NaiveKMeans(
    const MatType& dataset,
    DistanceType& distance) :
    dataset(dataset),
    distance(distance),
    distanceComputations(0)
{ /* Nothing to do. */ }

// Run a single iteration.
template<typename DistanceType, typename MatType, typename CentroidsType>
double NaiveKMeans<DistanceType, MatType, CentroidsType>::Iterate(
    const CentroidsType& centroids,
    CentroidsType& newCentroids,
    arma::Col<size_t>& counts)
{
  typedef typename MatType::elem_type ElemType;

  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Find the closest centroid to each point and update the new centroids.
  // Computed in parallel over the complete dataset
  #pragma omp parallel
  {
    // The current state of the K-means is private for each thread
    CentroidsType localCentroids(centroids.n_rows, centroids.n_cols);
    arma::Col<size_t> localCounts(centroids.n_cols);

    #pragma omp for schedule(static) nowait
    for (size_t i = 0; i < (size_t) dataset.n_cols; ++i)
    {
      // Find the closest centroid to this point.
      ElemType minDistance = std::numeric_limits<ElemType>::infinity();
      size_t closestCluster = centroids.n_cols; // Invalid value.

      for (size_t j = 0; j < centroids.n_cols; ++j)
      {
        const ElemType dist = distance.Evaluate(dataset.col(i),
            centroids.col(j));
        if (dist < minDistance)
        {
          minDistance = dist;
          closestCluster = j;
        }
      }

      Log::Assert(closestCluster != centroids.n_cols);

      // We now have the minimum distance centroid index.  Update that centroid.
      localCentroids.col(closestCluster) += dataset.col(i);
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
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < centroids.n_cols; ++i)
    if (counts(i) != 0)
      newCentroids.col(i) /= counts(i);

  distanceComputations += centroids.n_cols * dataset.n_cols;

  // Calculate cluster distortion for this iteration.
  ElemType cNorm = 0;
  #pragma omp parallel for reduction(+:cNorm) schedule(static)
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    cNorm += std::pow(distance.Evaluate(centroids.col(i), newCentroids.col(i)),
        2.0);
  }
  distanceComputations += centroids.n_cols;

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif
