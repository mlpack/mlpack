/**
 * @file methods/kmeans/hamerly_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of Greg Hamerly's algorithm for k-means clustering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_HPP
#define MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_HPP

namespace mlpack {

template<typename DistanceType, typename MatType>
class HamerlyKMeans
{
 public:
  /**
   * Construct the HamerlyKMeans object, which must store several sets of
   * bounds.
   */
  HamerlyKMeans(const MatType& dataset, DistanceType& metric);

  /**
   * Run a single iteration of Hamerly's algorithm, updating the given centroids
   * into the newCentroids matrix.
   *
   * @param centroids Current cluster centroids.
   * @param newCentroids New cluster centroids.
   * @param counts Current counts, to be overwritten with new counts.
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

  //! Minimum cluster distances from each cluster.
  arma::vec minClusterDistances;

  //! Upper bounds for each point.
  arma::vec upperBounds;
  //! Lower bounds for each point.
  arma::vec lowerBounds;
  //! Assignments for each point.
  arma::Col<size_t> assignments;

  //! Track distance calculations.
  size_t distanceCalculations;
};

} // namespace mlpack

// Include implementation.
#include "hamerly_kmeans_impl.hpp"

#endif
