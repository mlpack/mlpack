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
  const size_t dims = dataset.n_rows;
  const size_t points = dataset.n_cols;
  const size_t clusters = centroids.n_cols;

  newCentroids.zeros(dims, clusters);
  counts.zeros(clusters);

  // Pre-compute squared norms of centroids
  arma::vec centroidNorms(clusters);
  #pragma omp parallel for schedule(static)
  for (size_t j = 0; j < clusters; ++j)
  {
    centroidNorms(j) = arma::dot(centroids.col(j), centroids.col(j));
  }

  // Block-wise for a better cache utilization
  const size_t blockSize = 256; 
  
  #pragma omp parallel
  {
    arma::mat localCentroids(dims, clusters, arma::fill::zeros);
    arma::Col<size_t> localCounts(clusters, arma::fill::zeros);

    #pragma omp for schedule(dynamic, 1)
    for (size_t block = 0; block < points; block += blockSize)
    {
      const size_t blockEnd = std::min(block + blockSize, points);
      
      for (size_t i = block; i < blockEnd; ++i)
      {
        double minDistance = std::numeric_limits<double>::max();
        size_t closestCluster = clusters;

        const double* dataPoint = dataset.colptr(i);
        double dataNorm = 0.0;
        
        // Compute data point norm
        #pragma omp simd reduction(+:dataNorm)
        for (size_t d = 0; d < dims; ++d)
        {
          dataNorm += dataPoint[d] * dataPoint[d];
        }

        // Find closest centroid
        for (size_t j = 0; j < clusters; ++j)
        {
          const double* centroid = centroids.colptr(j);
          double dotProduct = 0.0;

          // Compute dot product
          #pragma omp simd reduction(+:dotProduct)
          for (size_t d = 0; d < dims; ++d)
          {
            dotProduct += dataPoint[d] * centroid[d];
          }

          // Use squared Euclidean distance
          double dist = dataNorm + centroidNorms(j) - 2 * dotProduct;

          if (dist < minDistance)
          {
            minDistance = dist;
            closestCluster = j;
          }
        }

        // Update local centroids and counts
        double* localCentroidCol = localCentroids.colptr(closestCluster);
        #pragma omp simd
        for (size_t d = 0; d < dims; ++d)
        {
          localCentroidCol[d] += dataPoint[d];
        }
        localCounts(closestCluster)++;
      }
    }

    // Combine results
    #pragma omp critical
    {
      newCentroids += localCentroids;
      counts += localCounts;
    }
  }

  // Normalize the centroids
  #pragma omp parallel for schedule(static)
  for (size_t j = 0; j < clusters; ++j)
  {
    if (counts(j) > 0)
    {
      newCentroids.col(j) /= counts(j);
    }
  }

  // Calculate cluster distortion
  double cNorm = 0.0;
  #pragma omp parallel for reduction(+:cNorm) schedule(static)
  for (size_t j = 0; j < clusters; ++j)
  {
    cNorm += arma::norm(centroids.col(j) - newCentroids.col(j), 2);
  }

  distanceCalculations += clusters * points;

  return cNorm;
}

} // namespace mlpack

#endif