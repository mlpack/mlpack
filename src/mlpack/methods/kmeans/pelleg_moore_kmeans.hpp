/**
 * @file methods/kmeans/pelleg_moore_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of Pelleg-Moore's 'blacklist' algorithm for k-means
 * clustering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_HPP
#define MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include "pelleg_moore_kmeans_statistic.hpp"

namespace mlpack {

/**
 * An implementation of Pelleg-Moore's 'blacklist' algorithm for k-means
 * clustering.  This algorithm builds a kd-tree on the data points and traverses
 * it in order to determine the closest clusters to each point.
 *
 * For more information on the algorithm, see
 *
 * @code
 * @inproceedings{pelleg1999accelerating,
 *     title={Accelerating exact k-means algorithms with geometric reasoning},
 *     author={Pelleg, Dan and Moore, Andrew W.},
 *     booktitle={Proceedings of the Fifth ACM SIGKDD International Conference
 *       on Knowledge Discovery and Data Mining (KDD '99)},
 * pages={277--281},
 * year={1999},
 * organization={ACM}
 * }
 * @endcode
 */
template<typename DistanceType, typename MatType>
class PellegMooreKMeans
{
 public:
  /**
   * Construct the PellegMooreKMeans object, which must construct a tree.
   */
  PellegMooreKMeans(const MatType& dataset, DistanceType& distance);

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

  //! Return the number of distance calculations.
  size_t DistanceCalculations() const { return distanceCalculations; }
  //! Modify the number of distance calculations.
  size_t& DistanceCalculations() { return distanceCalculations; }

  //! Convenience typedef for the tree.
  using TreeType = KDTree<DistanceType, PellegMooreKMeansStatistic, MatType>;

 private:
  //! The original dataset reference.
  const MatType& datasetOrig; // Maybe not necessary.
  //! The tree built on the points.
  TreeType* tree;
  //! The dataset we are using.
  const MatType& dataset;
  //! The distance metric.
  DistanceType& distance;

  //! Track distance calculations.
  size_t distanceCalculations;
};

} // namespace mlpack

#include "pelleg_moore_kmeans_impl.hpp"

#endif
