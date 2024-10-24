/**
 * @file methods/kmeans/dual_tree_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of a Lloyd iteration which uses dual-tree nearest neighbor
 * search as a black box.  The conditions under which this will perform best are
 * probably limited to the case where k is close to the number of points in the
 * dataset, and the number of iterations of the k-means algorithm will be few.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_HPP
#define MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include "dual_tree_kmeans_statistic.hpp"

namespace mlpack {

/**
 * An algorithm for an exact Lloyd iteration which simply uses dual-tree
 * nearest-neighbor search to find the nearest centroid for each point in the
 * dataset.  The conditions under which this will perform best are probably
 * limited to the case where k is close to the number of points in the dataset,
 * and the number of iterations of the k-means algorithm will be few.
 */
template<
    typename DistanceType,
    typename MatType,
    template<typename TreeDistanceType,
             typename TreeStatType,
             typename TreeMatType>
        class TreeType = KDTree>
class DualTreeKMeans
{
 public:
  //! Convenience typedef.
  using Tree = TreeType<DistanceType, DualTreeKMeansStatistic, MatType>;

  template<typename TreeDistanceType,
           typename IgnoredStatType,
           typename TreeMatType>
  using NNSTreeType =
      TreeType<TreeDistanceType, DualTreeKMeansStatistic, TreeMatType>;

  /**
   * Construct the DualTreeKMeans object, which will construct a tree on the
   * points.
   */
  DualTreeKMeans(const MatType& dataset, DistanceType& distance);

  /**
   * Delete the tree constructed by the DualTreeKMeans object.
   */
  ~DualTreeKMeans();

  /**
   * Run a single iteration of the dual-tree nearest neighbor algorithm for
   * k-means, updating the given centroids into the newCentroids matrix.
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

 private:
  //! The original dataset reference.
  const MatType& datasetOrig; // Maybe not necessary.
  //! The tree built on the points.
  Tree* tree;
  //! The dataset we are using.
  const MatType& dataset;
  //! The distance metric.
  DistanceType distance;

  //! Track distance calculations.
  size_t distanceCalculations;
  //! Track iteration number.
  size_t iteration;

  //! Upper bounds on nearest centroid.
  arma::vec upperBounds;
  //! Lower bounds on second closest cluster distance for each point.
  arma::vec lowerBounds;
  //! Indicator of whether or not the point is pruned.
  std::vector<bool> prunedPoints;

  arma::Row<size_t> assignments;

  std::vector<bool> visited; // Was the point visited this iteration?

  arma::mat lastIterationCentroids; // For sanity checks.

  arma::vec clusterDistances; // The amount the clusters moved last iteration.

  arma::mat interclusterDistances; // Static storage for intercluster distances.

  //! Update the bounds in the tree before the next iteration.
  //! centroids is the current (not yet searched) centroids.
  void UpdateTree(Tree& node,
                  const arma::mat& centroids,
                  const double parentUpperBound = 0.0,
                  const double adjustedParentUpperBound = DBL_MAX,
                  const double parentLowerBound = DBL_MAX,
                  const double adjustedParentLowerBound = 0.0);

  //! Extract the centroids of the clusters.
  void ExtractCentroids(Tree& node,
                        arma::mat& newCentroids,
                        arma::Col<size_t>& newCounts,
                        const arma::mat& centroids);

  void CoalesceTree(Tree& node, const size_t child = 0);
  void DecoalesceTree(Tree& node);
};

//! Utility function for hiding children.  This actually does something, and is
//! called if the tree is not a binary tree.
template<typename TreeType>
void HideChild(TreeType& node,
               const size_t child,
               const typename std::enable_if_t<
                   !TreeTraits<TreeType>::BinaryTree>* junk = 0);

//! Utility function for hiding children.  This is called when the tree is a
//! binary tree, and does nothing, because we don't hide binary children in this
//! way.
template<typename TreeType>
void HideChild(TreeType& node,
               const size_t child,
               const typename std::enable_if_t<
                   TreeTraits<TreeType>::BinaryTree>* junk = 0);

//! Utility function for restoring children to a non-binary tree.
template<typename TreeType>
void RestoreChildren(TreeType& node,
                     const typename std::enable_if_t<!TreeTraits<
                         TreeType>::BinaryTree>* junk = 0);

//! Utility function for restoring children to a binary tree.
template<typename TreeType>
void RestoreChildren(TreeType& node,
                     const typename std::enable_if_t<TreeTraits<
                         TreeType>::BinaryTree>* junk = 0);

//! A template typedef for the DualTreeKMeans algorithm with the default tree
//! type (a kd-tree).
template<typename DistanceType, typename MatType>
using DefaultDualTreeKMeans = DualTreeKMeans<DistanceType, MatType>;

//! A template typedef for the DualTreeKMeans algorithm with the cover tree
//! type.
template<typename DistanceType, typename MatType>
using CoverTreeDualTreeKMeans = DualTreeKMeans<DistanceType, MatType,
    StandardCoverTree>;

} // namespace mlpack

#include "dual_tree_kmeans_impl.hpp"

#endif
