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

#ifdef MLPACK_USE_OPENMP
  #include <omp.h>
#endif

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
  #ifdef MLPACK_USE_OPENMP
    #pragma omp parallel for reduction(+:distanceCalculations)
  #endif
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    for (size_t j = i + 1; j < centroids.n_cols; ++j)
    {
      const double dist = distance.Evaluate(centroids.col(i), centroids.col(j));
      ++distanceCalculations;

      if (dist > eps)
      {
        const double halfDist = dist / 2.0;
        // Update bounds, if this intra-cluster distance is smaller.
        #ifdef MLPACK_USE_OPENMP
          #pragma omp critical
        #endif
        {
          minClusterDistances(i) = std::min(minClusterDistances(i), halfDist);
          minClusterDistances(j) = std::min(minClusterDistances(j), halfDist);
        }
      }
    }
  }

  #ifdef MLPACK_USE_OPENMP
    #pragma omp parallel for reduction(+:distanceCalculations, hamerlyPruned)
  #endif
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    const double m = std::max(minClusterDistances(assignments[i]),
                              lowerBounds(i));

    // First bound test.
    if (upperBounds(i) <= m + eps)
    {
      ++hamerlyPruned;
      #ifdef MLPACK_USE_OPENMP
        #pragma omp critical
      #endif
      {
        newCentroids.col(assignments[i]) += dataset.col(i);
        ++counts(assignments[i]);
      }
      continue;
    }

    // Tighten upper bound.
    upperBounds(i) = distance.Evaluate(dataset.col(i),
                                       centroids.col(assignments[i]));
    ++distanceCalculations;

    // Second bound test.
    if (upperBounds(i) <= m + eps)
    {
      #ifdef MLPACK_USE_OPENMP
        #pragma omp critical
      #endif
      {
        newCentroids.col(assignments[i]) += dataset.col(i);
        ++counts(assignments[i]);
      }
      continue;
    }

    // The bounds failed. So test against all other clusters.
    // This is Hamerly's Point-All-Ctrs() function from the paper.
    // We have to reset the lower bound first.
    lowerBounds(i) = DBL_MAX;
    size_t newAssignment = assignments[i];
    double newUpperBound = upperBounds(i);
    double newLowerBound = DBL_MAX;

    for (size_t c = 0; c < centroids.n_cols; ++c)
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
    distanceCalculations += centroids.n_cols - 1;

    // Update bounds and assignment
    upperBounds(i) = newUpperBound;
    lowerBounds(i) = newLowerBound;
    assignments[i] = newAssignment;

    // Update new centroids.
    #ifdef MLPACK_USE_OPENMP
      #pragma omp critical
    #endif
    {
      newCentroids.col(newAssignment) += dataset.col(i);
      ++counts(newAssignment);
    }
  }

  // Normalize centroids and calculate cluster movement
  double furthestMovement = 0.0;
  double secondFurthestMovement = 0.0;
  size_t furthestMovingCluster = 0;
  arma::vec centroidMovements(centroids.n_cols);
  double centroidMovement = 0.0;

  #ifdef MLPACK_USE_OPENMP
    #pragma omp parallel for reduction(+:distanceCalculations, centroidMovement) \
                             reduction(max:furthestMovement)
  #endif
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts(c) > 0)
      newCentroids.col(c) /= counts(c);
    else
      newCentroids.col(c) = centroids.col(c);

    // Calculate movement.
    const double movement = std::sqrt(arma::sum(arma::square(centroids.col(c) - newCentroids.col(c))));
    centroidMovements(c) = movement;
    centroidMovement += std::pow(movement, 2.0);
    ++distanceCalculations;

    if (movement > furthestMovement)
    {
      #ifdef MLPACK_USE_OPENMP
        #pragma omp critical
      #endif
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
  #ifdef MLPACK_USE_OPENMP
    #pragma omp parallel for
  #endif
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (assignments[i] < centroids.n_cols)
    {
      upperBounds(i) += centroidMovements(assignments[i]);
      if (assignments[i] == furthestMovingCluster)
        lowerBounds(i) -= secondFurthestMovement;
      else
        lowerBounds(i) -= furthestMovement;
    }
    else
    {
      // Handle invalid assignment
      #ifdef MLPACK_USE_OPENMP
        #pragma omp critical
      #endif
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