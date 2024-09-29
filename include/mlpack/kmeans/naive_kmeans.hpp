/**
 * @file methods/kmeans/naive_kmeans.hpp
 * @author Ryan Curtin
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
#ifndef MLPACK_METHODS_KMEANS_NAIVE_KMEANS_HPP
#define MLPACK_METHODS_KMEANS_NAIVE_KMEANS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This is an implementation of a single iteration of Lloyd's algorithm for
 * k-means.  If your intention is to run the full k-means algorithm, you are
 * looking for the KMeans class instead of this one.  This class is used by
 * KMeans as the actual implementation of the Lloyd iteration.
 *
 * @param DistanceType Type of distance metric used with this implementation.
 * @param MatType Matrix type (arma::mat or arma::sp_mat).
 */
template<typename DistanceType, typename MatType>
class NaiveKMeans
{
 public:
  /**
   * Construct the NaiveKMeans object with the given dataset and distance
   * metric.
   *
   * @param dataset Dataset.
   * @param distance Instantiated distance metric.
   */
  NaiveKMeans(const MatType& dataset, DistanceType& distance);

  /**
   * Run a single iteration of the Lloyd algorithm, updating the given centroids
   * into the newCentroids matrix.  If any cluster is empty (that is, if any
   * cluster has no points assigned to it), then the centroid associated with
   * that cluster may be filled with invalid data (it will be corrected later).
   *
   * @param centroids Current cluster centroids.
   * @param newCentroids New cluster centroids.
   * @param counts Number of points in each cluster at the end of the iteration.
   */
  double Iterate(const arma::mat& centroids,
                 arma::mat& newCentroids,
                 arma::Col<size_t>& counts);

  size_t DistanceCalculations() const { return distanceCalculations; }

 private:
  //! The dataset.
  const MatType& dataset;
  //! The instantiated distance metric.
  DistanceType& distance;

  //! Number of distance calculations.
  size_t distanceCalculations;
};

} // namespace mlpack

// Include implementation.
#include "naive_kmeans_impl.hpp"

#endif
