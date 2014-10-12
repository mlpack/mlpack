/**
 * @file pelleg_moore_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of Pelleg-Moore's 'blacklist' algorithm for k-means
 * clustering.
 */
#ifndef __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include "pelleg_moore_kmeans_statistic.hpp"

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename MatType>
class PellegMooreKMeans
{
 public:
  /**
   * Construct the PellegMooreKMeans object, which must construct a tree.
   */
  PellegMooreKMeans(const MatType& dataset, MetricType& metric);

  /**
   * Delete the tree constructed by the PellegMooreKMeans object.
   */
  ~PellegMooreKMeans();

  /**
   * Run a single iteration of the Pelleg-Moore blacklist algorithm, updating
   * the given centroids into the newCentroids matrix.
   *
   * @param centroids Current cluster centroids.
   * @param newCentroids New cluster centroids.
   * @param counts Current counts, to be overwritten with new counts.
   */
  double Iterate(const arma::mat& centroids,
                 arma::mat& newCentroids,
                 arma::Col<size_t>& counts);

  size_t DistanceCalculations() const { return distanceCalculations; }

  typedef tree::BinarySpaceTree<bound::HRectBound<2, true>,
      PellegMooreKMeansStatistic, MatType> TreeType;

 private:
  //! The original dataset reference.
  const MatType& datasetOrig; // Maybe not necessary.
  //! The dataset we are using.
  const MatType& dataset;
  //! A copy of the dataset, if necessary.
  MatType datasetCopy;
  //! The metric.
  MetricType& metric;

  //! The tree built on the points.
  TreeType* tree;

  //! Track distance calculations.
  size_t distanceCalculations;
};

} // namespace kmeans
} // namespace mlpack

#include "pelleg_moore_kmeans_impl.hpp"

#endif
