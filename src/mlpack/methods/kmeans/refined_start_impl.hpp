/**
 * @file refined_start_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of Bradley and Fayyad's "Refining Initial Points for
 * K-Means clustering".  This class is meant to provide better initial points
 * for the k-means algorithm.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_KMEANS_REFINED_START_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_REFINED_START_IMPL_HPP

// In case it hasn't been included yet.
#include "refined_start.hpp"

namespace mlpack {
namespace kmeans {

//! Partition the given dataset according to Bradley and Fayyad's algorithm.
template<typename MatType>
void RefinedStart::Cluster(const MatType& data,
                           const size_t clusters,
                           arma::Col<size_t>& assignments) const
{
  math::RandomSeed(std::time(NULL));

  // This will hold the sampled datasets.
  const size_t numPoints = size_t(percentage * data.n_cols);
  MatType sampledData(data.n_rows, numPoints);
  // vector<bool> is packed so each bool is 1 bit.
  std::vector<bool> pointsUsed(data.n_cols, false);
  arma::mat sampledCentroids(data.n_rows, samplings * clusters);

  // We will use these objects repeatedly for clustering.
  arma::Col<size_t> sampledAssignments;
  arma::mat centroids;
  KMeans<> kmeans;

  for (size_t i = 0; i < samplings; ++i)
  {
    // First, assemble the sampled dataset.
    size_t curSample = 0;
    while (curSample < numPoints)
    {
      // Pick a random point in [0, numPoints).
      size_t sample = (size_t) math::RandInt(data.n_cols);

      if (!pointsUsed[sample])
      {
        // This point isn't used yet.  So we'll put it in our sample.
        pointsUsed[sample] = true;
        sampledData.col(curSample) = data.col(sample);
        ++curSample;
      }
    }

    // Now, using the sampled dataset, run k-means.  In the case of an empty
    // cluster, we re-initialize that cluster as the point furthest away from
    // the cluster with maximum variance.  This is not *exactly* what the paper
    // implements, but it is quite similar, and we'll call it "good enough".
    kmeans.Cluster(sampledData, clusters, sampledAssignments, centroids);

    // Store the sampled centroids.
    sampledCentroids.cols(i * clusters, (i + 1) * clusters - 1) = centroids;

    pointsUsed.assign(data.n_cols, false);
  }

  // Now, we run k-means on the sampled centroids to get our final clusters.
  kmeans.Cluster(sampledCentroids, clusters, sampledAssignments, centroids);

  // Turn the final centroids into assignments.
  assignments.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Find the closest centroid to this point.
    double minDistance = std::numeric_limits<double>::infinity();
    size_t closestCluster = clusters;

    for (size_t j = 0; j < clusters; ++j)
    {
      const double distance = kmeans.Metric().Evaluate(data.col(i),
          centroids.col(j));

      if (distance < minDistance)
      {
        minDistance = distance;
        closestCluster = j;
      }
    }

    // Assign the point to its closest cluster.
    assignments[i] = closestCluster;
  }
}

}; // namespace kmeans
}; // namespace mlpack

#endif
