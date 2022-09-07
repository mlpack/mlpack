/**
 * @file methods/kmeans/refined_start_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of Bradley and Fayyad's "Refining Initial Points for
 * K-Means clustering".  This class is meant to provide better initial points
 * for the k-means algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_REFINED_START_IMPL_HPP
#define MLPACK_METHODS_KMEANS_REFINED_START_IMPL_HPP

// In case it hasn't been included yet.
#include "refined_start.hpp"

namespace mlpack {

//! Partition the given dataset according to Bradley and Fayyad's algorithm.
template<typename MatType>
void RefinedStart::Cluster(const MatType& data,
                           const size_t clusters,
                           arma::mat& centroids) const
{
  // This will hold the sampled datasets.
  const size_t numPoints = size_t(percentage * data.n_cols);
  MatType sampledData(data.n_rows, numPoints);
  // vector<bool> is packed so each bool is 1 bit.
  std::vector<bool> pointsUsed(data.n_cols, false);
  arma::mat sampledCentroids(data.n_rows, samplings * clusters);

  for (size_t i = 0; i < samplings; ++i)
  {
    // First, assemble the sampled dataset.
    size_t curSample = 0;
    while (curSample < numPoints)
    {
      // Pick a random point in [0, numPoints).
      size_t sample = (size_t) RandInt(data.n_cols);

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
    KMeans<> kmeans;
    kmeans.Cluster(sampledData, clusters, centroids);

    // Store the sampled centroids.
    sampledCentroids.cols(i * clusters, (i + 1) * clusters - 1) = centroids;

    pointsUsed.assign(data.n_cols, false);
  }

  // Now, we run k-means on the sampled centroids to get our final clusters.
  KMeans<> kmeans;
  kmeans.Cluster(sampledCentroids, clusters, centroids);
}

template<typename MatType>
void RefinedStart::Cluster(const MatType& data,
                           const size_t clusters,
                           arma::Row<size_t>& assignments) const
{
  // Perform the Bradley-Fayyad refined start algorithm, and get initial
  // centroids back.
  arma::mat centroids;
  Cluster(data, clusters, centroids);

  // Turn the final centroids into assignments.
  assignments.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Find the closest centroid to this point.
    double minDistance = std::numeric_limits<double>::infinity();
    size_t closestCluster = clusters;

    for (size_t j = 0; j < clusters; ++j)
    {
      // This is restricted to the L2 distance, and unfortunately it would take
      // a lot of refactoring and redesign to make this more general... we would
      // probably need to have KMeans take a template template parameter for the
      // initial partition policy.  It's not clear how to best do this.
      const double distance = EuclideanDistance::Evaluate(data.col(i),
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

} // namespace mlpack

#endif
