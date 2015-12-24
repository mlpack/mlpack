/**
 * @file hamerly_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of Greg Hamerly's algorithm for k-means clustering.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_HPP

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename MatType>
class HamerlyKMeans
{
 public:
  /**
   * Construct the HamerlyKMeans object, which must store several sets of
   * bounds.
   */
  HamerlyKMeans(const MatType& dataset, MetricType& metric);

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
  //! The instantiated metric.
  MetricType& metric;

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

} // namespace kmeans
} // namespace mlpack

// Include implementation.
#include "hamerly_kmeans_impl.hpp"

#endif
