/**
 * @file naive_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of a naively-implemented step of the Lloyd algorithm for
 * k-means clustering.  This may still be the best choice for small datasets or
 * datasets with very high dimensionality.
 */
#ifndef __MLPACK_METHODS_KMEANS_NAIVE_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_NAIVE_KMEANS_HPP

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename MatType>
class NaiveKMeans
{
 public:
  NaiveKMeans(const MatType& dataset, MetricType& metric);

  /**
   * Run a single iteration of the Lloyd algorithm, updating the given centroids
   * into the newCentroids matrix.
   *
   * @param centroids Current cluster centroids.
   * @param newCentroids New cluster centroids.
   */
  void Iterate(const arma::mat& centroids,
               arma::mat& newCentroids,
               arma::Col<size_t>& counts);

 private:
  //! The dataset.
  const MatType& dataset;
  //! The instantiated metric.
  MetricType& metric;
};

} // namespace kmeans
} // namespace mlpack

// Include implementation.
#include "naive_kmeans_impl.hpp"

#endif
