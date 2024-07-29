/**
 * @file methods/kmeans/hamerly_kmeans_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of Greg Hamerly's algorithm for k-means clustering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_IMPL_HPP
#define MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_IMPL_HPP

// In case it hasn't been included yet.
#include "hamerly_kmeans.hpp"

namespace mlpack {

template<typename DistanceType, typename MatType>
HamerlyKMeans<DistanceType, MatType>::HamerlyKMeans(const MatType& dataset,
                                                    DistanceType& distance) :
    dataset(dataset),
    distance(distance),
    distanceCalculations(0)
{
  // Nothing to do.
}
template<typename DistanceType, typename MatType>
double HamerlyKMeans<DistanceType, MatType>::Iterate(const arma::mat& centroids,
                                                     arma::mat& newCentroids,
                                                     arma::Col<size_t>& counts)
{
  static constexpr double eps = std::numeric_limits<double>::epsilon();
  size_t hamerlyPruned = 0;

  // If this is the first iteration, we need to set all the bounds.
  if (minClusterDistances.n_elem != centroids.n_cols)
  {
    upperBounds.set_size(dataset.n_cols);
    upperBounds.fill(DBL_MAX);
    lowerBounds.zeros(dataset.n_cols);
    assignments.zeros(dataset.n_cols);
    minClusterDistances.set_size(centroids.n_cols);
  }

  // Reset new centroids and counts.
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Calculate minimum intra-cluster distance for each cluster.
  minClusterDistances.fill(DBL_MAX);
  
  #pragma omp parallel
  {
    arma::vec localMinClusterDistances(centroids.n_cols, arma::fill::value(DBL_MAX));
    #pragma omp for reduction(+:distanceCalculations) schedule(dynamic)
    for (size_t i = 0; i < centroids.n_cols; ++i)
    {
      for (size_t j = i + 1; j < centroids.n_cols; ++j)
      {
        const double dist = distance.Evaluate(centroids.col(i), centroids.col(j));
        ++distanceCalculations;

        if (dist > eps)
        {
          const double halfDist = dist / 2.0;
          localMinClusterDistances(i) = std::min(localMinClusterDistances(i), halfDist);
          localMinClusterDistances(j) = std::min(localMinClusterDistances(j), halfDist);
        }
      }
    }

    #pragma omp critical
    {
      for (size_t i = 0; i < centroids.n_cols; ++i)
      {
        minClusterDistances(i) = std::min(minClusterDistances(i), localMinClusterDistances(i));
      }
    }
  }

  const size_t numPoints = dataset.n_cols;
  const size_t numClusters = centroids.n_cols;

  #pragma omp parallel
  {
    arma::mat localNewCentroids(centroids.n_rows, centroids.n_cols, arma::fill::zeros);
    arma::Col<size_t> localCounts(centroids.n_cols, arma::fill::zeros);
    size_t localHamerlyPruned = 0;
    size_t localDistanceCalculations = 0;

    #pragma omp for schedule(static)
    for (size_t i = 0; i < numPoints; ++i)
    {
      const double m = std::max(minClusterDistances(assignments[i]),
                                lowerBounds(i));

      // First bound test.
      if (upperBounds(i) <= m + eps)
      {
        ++localHamerlyPruned;
        localNewCentroids.col(assignments[i]) += dataset.col(i);
        ++localCounts(assignments[i]);
        continue;
      }

      // Tighten upper bound.
      upperBounds(i) = distance.Evaluate(dataset.col(i),
                                         centroids.col(assignments[i]));
      ++localDistanceCalculations;

      // Second bound test.
      if (upperBounds(i) <= m + eps)
      {
        localNewCentroids.col(assignments[i]) += dataset.col(i);
        ++localCounts(assignments[i]);
        continue;
      }

      // The bounds failed. So test against all other clusters.
      lowerBounds(i) = DBL_MAX;
      size_t newAssignment = assignments[i];
      double newUpperBound = upperBounds(i);
      double newLowerBound = DBL_MAX;

      for (size_t c = 0; c < numClusters; ++c)
      {
        if (c == assignments[i])
          continue;

        const double dist = distance.Evaluate(dataset.col(i), centroids.col(c));

        // Is this a better cluster?
        if (dist < newUpperBound)
        {
          newLowerBound = newUpperBound;
          newUpperBound = dist;
          newAssignment = c;
        }
        else if (dist < newLowerBound)
        {
          newLowerBound = dist;
        }
      }
      localDistanceCalculations += numClusters - 1;

      // Update bounds and assignment
      upperBounds(i) = newUpperBound;
      lowerBounds(i) = newLowerBound;
      assignments[i] = newAssignment;

      // Update new centroids.
      localNewCentroids.col(newAssignment) += dataset.col(i);
      ++localCounts(newAssignment);
    }

    #pragma omp critical
    {
      newCentroids += localNewCentroids;
      counts += localCounts;
      hamerlyPruned += localHamerlyPruned;
      distanceCalculations += localDistanceCalculations;
    }
  }

  // Normalize centroids and calculate cluster movement
  double furthestMovement = 0.0;
  double secondFurthestMovement = 0.0;
  size_t furthestMovingCluster = 0;
  arma::vec centroidMovements(numClusters);
  double centroidMovement = 0.0;

  #pragma omp parallel for reduction(+:centroidMovement) \
      reduction(max:furthestMovement) schedule(static)
  for (size_t c = 0; c < numClusters; ++c)
  {
    if (counts(c) > 0)
      newCentroids.col(c) /= counts(c);
    else
      newCentroids.col(c) = centroids.col(c);

    // Calculate movement.
    const double movement = std::sqrt(arma::sum(arma::square(centroids.col(c) - newCentroids.col(c))));
    centroidMovements(c) = movement;
    centroidMovement += std::pow(movement, 2.0);

    if (movement > furthestMovement)
    {
      #pragma omp critical
      {
        if (movement > furthestMovement)
        {
          secondFurthestMovement = furthestMovement;
          furthestMovement = movement;
          furthestMovingCluster = c;
        }
        else if (movement > secondFurthestMovement)
        {
          secondFurthestMovement = movement;
        }
      }
    }
  }

  // Now update bounds
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < numPoints; ++i)
  {
    if (assignments[i] < numClusters)
    {
      upperBounds(i) += centroidMovements(assignments[i]);
      if (assignments[i] == furthestMovingCluster)
        lowerBounds(i) -= secondFurthestMovement;
      else
        lowerBounds(i) -= furthestMovement;
    }
    else
    {
      #pragma omp critical
      {
        Log::Warn << "Invalid assignment for point " << i << std::endl;
      }
    }
  }

  Log::Info << "Hamerly prunes: " << hamerlyPruned << ".\n";

  return std::sqrt(centroidMovement);
}
} // namespace mlpack

#endif