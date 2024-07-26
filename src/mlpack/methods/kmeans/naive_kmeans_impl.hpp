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

  // Pre-compute centroid norms if using Euclidean distance
  arma::vec centroidNorms;
  if (std::is_same<DistanceType, EuclideanDistance>::value)
  {
    centroidNorms = arma::sum(arma::square(centroids), 0).t();
  }

  #pragma omp parallel
  {
    // Thread-local storage
    arma::mat localNewCentroids(dims, clusters, arma::fill::zeros);
    arma::Col<size_t> localCounts(clusters, arma::fill::zeros);

    #pragma omp for schedule(static)
    for (size_t i = 0; i < points; ++i)
    {
      size_t closestCluster = 0;
      double minDistance = std::numeric_limits<double>::max();
      const auto& point = dataset.col(i);

      if (std::is_same<DistanceType, EuclideanDistance>::value)
      {
        // Optimized Euclidean distance calculation
        const double pointNorm = arma::dot(point, point);
        for (size_t j = 0; j < clusters; ++j)
        {
          const double dist = pointNorm + centroidNorms[j] - 2 * arma::dot(point, centroids.col(j));
          if (dist < minDistance)
          {
            minDistance = dist;
            closestCluster = j;
          }
        }
      }
      else
      {
        // General distance metric
        for (size_t j = 0; j < clusters; ++j)
        {
          const double dist = distance.Evaluate(point, centroids.col(j));
          if (dist < minDistance)
          {
            minDistance = dist;
            closestCluster = j;
          }
        }
      }

      // Update local centroids and counts
      localNewCentroids.col(closestCluster) += point;
      localCounts[closestCluster]++;
    }

    // Combine results
    #pragma omp critical
    {
      newCentroids += localNewCentroids;
      counts += localCounts;
    }
  }

  // Normalize the centroids
  for (size_t j = 0; j < clusters; ++j)
  {
    if (counts[j] > 0)
      newCentroids.col(j) /= counts[j];
    else
      newCentroids.col(j) = centroids.col(j);
  }

  // Calculate cluster distortion
  double cNorm = 0.0;
  #pragma omp parallel for reduction(+:cNorm) schedule(static)
  for (size_t j = 0; j < clusters; ++j)
  {
    cNorm += std::pow(arma::norm(centroids.col(j) - newCentroids.col(j)), 2);
  }

  distanceCalculations += clusters * points;

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif