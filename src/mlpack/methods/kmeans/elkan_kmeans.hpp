/**
 * @file methods/kmeans/elkan_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of Elkan's algorithm for exact Lloyd iterations.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_ELKAN_KMEANS_HPP
#define MLPACK_METHODS_KMEANS_ELKAN_KMEANS_HPP

namespace mlpack {

template<typename DistanceType, typename MatType, typename CentroidsType>
class ElkanKMeans
{
 public:
  typedef typename MatType::elem_type ElemType;
  typedef typename GetDenseColType<MatType>::type ColType;

  /**
   * Construct the ElkanKMeans object, which must store several sets of bounds.
   */
  ElkanKMeans(const MatType& dataset, DistanceType& distance);

  /**
   * Run a single iteration of Elkan's algorithm, updating the given centroids
   * into the newCentroids matrix.
   *
   * @param centroids Current cluster centroids.
   * @param newCentroids New cluster centroids.
   * @param counts Current counts, to be overwritten with new counts.
   */
  double Iterate(const CentroidsType& centroids,
                 CentroidsType& newCentroids,
                 arma::Col<size_t>& counts);

  size_t DistanceCalculations() const { return distanceCalculations; }

 private:
  // The dataset.
  const MatType& dataset;
  // The instantiated distance metric.
  DistanceType& distance;

  // Holds intra-cluster distances.
  CentroidsType clusterDistances;
  // Half the distance from a cluster to its nearest cluster (s(c)).
  ColType minClusterDistances;

  // Holds the index of the cluster that owns each point.
  arma::Col<size_t> assignments;

  // Upper bounds on the distance between each point and its closest cluster.
  CentroidsType upperBounds;
  // Lower bounds on the distance between each point and each cluster.
  CentroidsType lowerBounds;

  // Track distance calculations.
  size_t distanceCalculations;
};

} // namespace mlpack

// Include implementation.
#include "elkan_kmeans_impl.hpp"

#endif
