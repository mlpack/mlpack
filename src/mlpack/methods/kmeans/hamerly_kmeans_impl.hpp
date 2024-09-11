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

  // Reset new centroids.
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Calculate minimum intra-cluster distance for each cluster.
  minClusterDistances.fill(DBL_MAX);
  #pragma omp parallel for reduction(+:distanceCalculations) schedule(static)
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    for (size_t j = i + 1; j < centroids.n_cols; ++j)
    {
      const double dist = distance.Evaluate(centroids.col(i),
                                            centroids.col(j)) / 2.0;
      ++distanceCalculations;

      // Update bounds, if this intra-cluster distance is smaller.
      minClusterDistances(i) = std::min(minClusterDistances(i), dist);
      minClusterDistances(j) = std::min(minClusterDistances(j), dist);
    }
  }

  #pragma omp parallel for reduction(+:hamerlyPruned, distanceCalculations) \
      reduction(matAdd:newCentroids) reduction(colAdd:counts) schedule(static)
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    const double m = std::max(minClusterDistances(assignments[i]),
                              lowerBounds(i));

    // First bound test.
    if (upperBounds(i) <= m)
    {
      ++hamerlyPruned;
      newCentroids.col(assignments[i]) += dataset.col(i);
      ++counts(assignments[i]);
      continue;
    }

    // Tighten upper bound.
    upperBounds(i) = distance.Evaluate(dataset.col(i),
                                       centroids.col(assignments[i]));
    ++distanceCalculations;

    // Second bound test.
    if (upperBounds(i) <= m)
    {
      newCentroids.col(assignments[i]) += dataset.col(i);
      ++counts(assignments[i]);
      continue;
    }

    // The bounds failed.  So test against all other clusters.
    // This is Hamerly's Point-All-Ctrs() function from the paper.
    // We have to reset the lower bound first.
    lowerBounds(i) = DBL_MAX;
    for (size_t c = 0; c < centroids.n_cols; ++c)
    {
      if (c == assignments[i])
        continue;

      const double dist = distance.Evaluate(dataset.col(i), centroids.col(c));

      // Is this a better cluster?  At this point, upperBounds[i] = d(i, c(i)).
      if (dist < upperBounds(i))
      {
        // lowerBounds holds the second closest cluster.
        lowerBounds(i) = upperBounds(i);
        upperBounds(i) = dist;
        assignments[i] = c;
      }
      else if (dist < lowerBounds(i))
      {
        // This is a closer second-closest cluster.
        lowerBounds(i) = dist;
      }
    }
    distanceCalculations += centroids.n_cols - 1;

    // Update new centroids.
    newCentroids.col(assignments[i]) += dataset.col(i);
    ++counts(assignments[i]);
  }

  // Normalize centroids and calculate cluster movement (contains parts of
  // Move-Centers() and Update-Bounds()).
  double furthestMovement = 0.0;
  double secondFurthestMovement = 0.0;
  size_t furthestMovingCluster = 0;
  arma::vec centroidMovements(centroids.n_cols);
  double centroidMovement = 0.0;
  #pragma omp parallel for \
      reduction(+: distanceCalculations, centroidMovement) schedule(static)
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts(c) > 0)
      newCentroids.col(c) /= counts(c);

    // Calculate movement.
    const double movement = distance.Evaluate(centroids.col(c),
                                              newCentroids.col(c));
    centroidMovements(c) = movement;
    centroidMovement += std::pow(movement, 2.0);
    ++distanceCalculations;

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

  // Now update bounds (lines 3-8 of Update-Bounds()).
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    upperBounds(i) += centroidMovements(assignments[i]);
    if (assignments[i] == furthestMovingCluster)
      lowerBounds(i) -= secondFurthestMovement;
    else
      lowerBounds(i) -= furthestMovement;
  }

  Log::Info << "Hamerly prunes: " << hamerlyPruned << ".\n";

  return std::sqrt(centroidMovement);
}

} // namespace mlpack

#endif
