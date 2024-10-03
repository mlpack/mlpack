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

// Run a single iteration of Elkan's algorithm for Lloyd iterations.
template<typename DistanceType, typename MatType>
double ElkanKMeans<DistanceType, MatType>::Iterate(const arma::mat& centroids,
                                                   arma::mat& newCentroids,
                                                   arma::Col<size_t>& counts)
{
  // Clear new centroids.
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // At the beginning of the iteration, we must compute the distances between
  // all centers.  This is O(k^2).
  clusterDistances.set_size(centroids.n_cols, centroids.n_cols);

  // Self-distances are always 0, but we set them to DBL_MAX to avoid the self
  // being the closest cluster centroid.
  clusterDistances.diag().fill(DBL_MAX);

  // Initially set r(x) to true.
  std::vector<bool> mustRecalculate(dataset.n_cols, true);

  // If this is the first iteration, we must reset all the bounds.
  if (lowerBounds.n_rows != centroids.n_cols)
  {
    lowerBounds.set_size(centroids.n_cols, dataset.n_cols);
    assignments.set_size(dataset.n_cols);
    upperBounds.set_size(dataset.n_cols);

    lowerBounds.fill(0);
    upperBounds.fill(DBL_MAX);
    assignments.fill(0);
  }

  // Step 1: for all centers, compute between-cluster distances.  For all
  // centers, compute s(c) = 1/2 min d(c, c').
  #pragma omp parallel for schedule(dynamic) reduction(+:distanceCalculations)
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    for (size_t j = i + 1; j < centroids.n_cols; ++j)
    {
      const double dist = distance.Evaluate(centroids.col(i),
                                            centroids.col(j));
      distanceCalculations++;
      clusterDistances(i, j) = dist;
      clusterDistances(j, i) = dist;
    }
  }

  // Now find the closest cluster to each other cluster.  We multiply by 0.5 so
  // that this is equivalent to s(c) for each cluster c.
  minClusterDistances = 0.5 * min(clusterDistances).t();

  // Now loop over all points, and see which ones need to be updated.
  #pragma omp parallel for schedule(dynamic) reduction(matAdd: newCentroids) \
      reduction(colAdd: counts) reduction(+: distanceCalculations)
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    // Step 2: identify all points such that u(x) <= s(c(x)).
    if (upperBounds(i) <= minClusterDistances(assignments[i]))
    {
      // No change needed.  This point must still belong to that cluster.
      counts(assignments[i])++;
      newCentroids.col(assignments[i]) += arma::vec(dataset.col(i));
    }
    else
    {
      for (size_t c = 0; c < centroids.n_cols; ++c)
      {
        // Step 3: for all remaining points x and centers c such that c != c(x),
        // u(x) > l(x, c) and u(x) > 0.5 d(c(x), c)...
        if (assignments[i] == c)
          continue; // Pruned because this cluster is already the assignment.

        if (upperBounds(i) <= lowerBounds(c, i))
          continue; // Pruned by triangle inequality on lower bound.

        if (upperBounds(i) <= 0.5 * clusterDistances(assignments[i], c))
          continue; // Pruned by triangle inequality on cluster distances.

        // Step 3a: if r(x) then compute d(x, c(x)) and assign r(x) = false.
        // Otherwise, d(x, c(x)) = u(x).
        double dist;
        if (mustRecalculate[i])
        {
          mustRecalculate[i] = false;
          dist = distance.Evaluate(dataset.col(i),
                                   centroids.col(assignments[i]));
          lowerBounds(assignments[i], i) = dist;
          upperBounds(i) = dist;
          distanceCalculations++;

          // Check if we can prune again.
          if (upperBounds(i) <= lowerBounds(c, i))
            continue; // Pruned by triangle inequality on lower bound.

          if (upperBounds(i) <= 0.5 * clusterDistances(assignments[i], c))
            continue; // Pruned by triangle inequality on cluster distances.
        }
        else
        {
          dist = upperBounds(i); // This is equivalent to d(x, c(x)).
        }

        // Step 3b: if d(x, c(x)) > l(x, c) or d(x, c(x)) > 0.5 d(c(x), c)...
        if (dist > lowerBounds(c, i) ||
            dist > 0.5 * clusterDistances(assignments[i], c))
        {
          // Compute d(x, c).  If d(x, c) < d(x, c(x)) then assign c(x) = c.
          const double pointDist = distance.Evaluate(dataset.col(i),
                                                     centroids.col(c));
          lowerBounds(c, i) = pointDist;
          distanceCalculations++;
          if (pointDist < dist)
          {
            upperBounds(i) = pointDist;
            assignments[i] = c;
          }
        }
      }

      // At this point, we know the new cluster assignment.
      // Step 4: for each center c, let m(c) be the mean of the points
      // assigned to c.
      newCentroids.col(assignments[i]) += arma::vec(dataset.col(i));
      counts[assignments[i]]++;
    }
  }

  // Now, normalize and calculate the distance each cluster has moved.
  arma::vec moveDistances(centroids.n_cols);
  double cNorm = 0.0; // Cluster movement for residual.
  #pragma omp parallel for reduction(+: cNorm, distanceCalculations)
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts[c] > 0)
      newCentroids.col(c) /= counts[c];

    moveDistances(c) = distance.Evaluate(newCentroids.col(c), centroids.col(c));
    cNorm += std::pow(moveDistances(c), 2.0);
    distanceCalculations++;
  }

  #pragma omp parallel for
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    // Step 5: for each point x and center c, assign
    //   l(x, c) = max { l(x, c) - d(c, m(c)), 0 }.
    // But it doesn't actually matter if l(x, c) is positive.
    for (size_t c = 0; c < centroids.n_cols; ++c)
      lowerBounds(c, i) -= moveDistances(c);

    // Step 6: for each point x, assign
    //   u(x) = u(x) + d(m(c(x)), c(x))
    //   r(x) = true (we are setting that at the start of every iteration).
    upperBounds(i) += moveDistances(assignments[i]);
  }

  return std::sqrt(cNorm);
}

} // namespace mlpack

#endif
