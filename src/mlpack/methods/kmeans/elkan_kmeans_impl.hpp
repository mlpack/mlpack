/**
 * @file methods/kmeans/elkan_kmeans_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of Elkan's algorithm for exact Lloyd iterations.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_ELKAN_KMEANS_IMPL_HPP
#define MLPACK_METHODS_KMEANS_ELKAN_KMEANS_IMPL_HPP

// In case it hasn't been included yet.
#include "elkan_kmeans.hpp"

namespace mlpack {
template<typename DistanceType, typename MatType>
ElkanKMeans<DistanceType, MatType>::ElkanKMeans(const MatType& dataset,
                                                DistanceType& distance) :
    dataset(dataset),
    distance(distance),
    distanceCalculations(0)
{
  // Nothing to do here.
}

template<typename DistanceType, typename MatType>
double ElkanKMeans<DistanceType, MatType>::Iterate(const arma::mat& centroids,
                                                   arma::mat& newCentroids,
                                                   arma::Col<size_t>& counts)
{
  // Clear new centroids and counts.
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Compute the distances between all centers (O(k^2)).
  clusterDistances.set_size(centroids.n_cols, centroids.n_cols);
  clusterDistances.diag().fill(DBL_MAX);

  #pragma omp parallel for schedule(dynamic) reduction(+:distanceCalculations)
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    for (size_t j = i + 1; j < centroids.n_cols; ++j)
    {
      const double dist = distance.Evaluate(centroids.col(i), centroids.col(j));
      clusterDistances(i, j) = dist;
      clusterDistances(j, i) = dist;
      distanceCalculations++;
    }
  }

  // Find the closest cluster to each other cluster.
  minClusterDistances = 0.5 * min(clusterDistances).t();

  // Initialize or reset bounds if necessary
  if (lowerBounds.n_rows != centroids.n_cols)
  {
    lowerBounds.set_size(centroids.n_cols, dataset.n_cols);
    assignments.set_size(dataset.n_cols);
    upperBounds.set_size(dataset.n_cols);

    lowerBounds.fill(0);
    upperBounds.fill(DBL_MAX);
    assignments.fill(0);
  }

  // Determine the number of threads
  int numThreads;
  #pragma omp parallel
  {
    #pragma omp single
    numThreads = omp_get_num_threads();
  }

  // Create thread-local storage for newCentroids and counts
  std::vector<arma::mat> threadNewCentroids(numThreads, arma::mat(centroids.n_rows, centroids.n_cols, arma::fill::zeros));
  std::vector<arma::Col<size_t>> threadCounts(numThreads, arma::Col<size_t>(centroids.n_cols, arma::fill::zeros));

  // Main loop over all points
  #pragma omp parallel for schedule(dynamic) reduction(+:distanceCalculations)
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    int threadId = omp_get_thread_num();
    bool mustRecalculate = true;

    if (upperBounds(i) <= minClusterDistances(assignments[i]))
    {
      // No change needed. This point must still belong to that cluster.
      threadCounts[threadId](assignments[i])++;
      threadNewCentroids[threadId].col(assignments[i]) += arma::vec(dataset.col(i));
      continue;
    }

    for (size_t c = 0; c < centroids.n_cols; ++c)
    {
      if (assignments[i] == c) continue;
      if (upperBounds(i) <= lowerBounds(c, i)) continue;
      if (upperBounds(i) <= 0.5 * clusterDistances(assignments[i], c)) continue;

      double dist;
      if (mustRecalculate)
      {
        mustRecalculate = false;
        dist = distance.Evaluate(dataset.col(i), centroids.col(assignments[i]));
        lowerBounds(assignments[i], i) = dist;
        upperBounds(i) = dist;
        distanceCalculations++;

        if (upperBounds(i) <= lowerBounds(c, i)) continue;
        if (upperBounds(i) <= 0.5 * clusterDistances(assignments[i], c)) continue;
      }
      else
      {
        dist = upperBounds(i);
      }

      if (dist > lowerBounds(c, i) || dist > 0.5 * clusterDistances(assignments[i], c))
      {
        const double pointDist = distance.Evaluate(dataset.col(i), centroids.col(c));
        lowerBounds(c, i) = pointDist;
        distanceCalculations++;
        if (pointDist < dist)
        {
          upperBounds(i) = pointDist;
          assignments[i] = c;
        }
      }
    }

    threadNewCentroids[threadId].col(assignments[i]) += arma::vec(dataset.col(i));
    threadCounts[threadId][assignments[i]]++;
  }

  // Combine thread-local results
  for (int t = 0; t < numThreads; ++t)
  {
    newCentroids += threadNewCentroids[t];
    counts += threadCounts[t];
  }

  // Normalize and calculate the distance each cluster has moved.
  arma::vec moveDistances(centroids.n_cols);
  double cNorm = 0.0;

  #pragma omp parallel for reduction(+:cNorm,distanceCalculations)
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts[c] > 0)
      newCentroids.col(c) /= counts[c];

    moveDistances(c) = distance.Evaluate(newCentroids.col(c), centroids.col(c));
    cNorm += std::pow(moveDistances(c), 2.0);
    distanceCalculations++;
  }

  // Update bounds
  #pragma omp parallel for
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (size_t c = 0; c < centroids.n_cols; ++c)
      lowerBounds(c, i) -= moveDistances(c);

    upperBounds(i) += moveDistances(assignments[i]);
  }

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif
