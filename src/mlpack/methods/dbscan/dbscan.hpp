/**
 * @file methods/dbscan/dbscan.hpp
 * @author Ryan Curtin
 *
 * An implementation of the DBSCAN clustering method, which is flexible enough
 * to support other algorithms for finding nearest neighbors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DBSCAN_DBSCAN_HPP
#define MLPACK_METHODS_DBSCAN_DBSCAN_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/range_search/range_search.hpp>
#include <mlpack/methods/emst/union_find.hpp>
#include "random_point_selection.hpp"
#include "ordered_point_selection.hpp"

namespace mlpack {

/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a
 * clustering technique described in the following paper:
 *
 * @code
 * @inproceedings{ester1996density,
 *   title={A density-based algorithm for discovering clusters in large spatial
 *       databases with noise.},
 *   author={Ester, M. and Kriegel, H.-P. and Sander, J. and Xu, X.},
 *   booktitle={Proceedings of the Second International Conference on Knowledge
 *       Discovery and Data Mining (KDD '96)},
 *   pages={226--231},
 *   year={1996}
 * }
 * @endcode
 *
 * The DBSCAN algorithm iteratively clusters points using range searches with a
 * specified radius parameter.  This implementation allows configuration of the
 * range search technique used and the point selection strategy by means of
 * template parameters.
 *
 * @tparam RangeSearchType Class to use for range searching.
 * @tparam PointSelectionPolicy Strategy for selecting next point to cluster
 *      with.
 */
template<typename RangeSearchType = RangeSearch<>,
         typename PointSelectionPolicy = OrderedPointSelection>
class DBSCAN
{
 public:
  //! Easy access to the MatType.
  using MatType = typename RangeSearchType::Mat;
  //! Easy access to Element Type of the matrix.
  using ElemType = typename RangeSearchType::Mat::elem_type;
  /**
   * Construct the DBSCAN object with the given parameters.  The batchMode
   * parameter should be set to false in the case where RAM issues will be
   * encountered (i.e. if the dataset is very large or if epsilon is large).
   * When batchMode is false, each point will be searched iteratively, which
   * could be slower but will use less memory.
   *
   * @param epsilon Size of range query.
   * @param minPoints Minimum number of points for each cluster.
   * @param batchMode If true, all points are searched in batch.
   * @param rangeSearch Optional instantiated RangeSearch object.
   * @param pointSelector OptionL instantiated PointSelectionPolicy object.
   */
  DBSCAN(const ElemType epsilon,
         const size_t minPoints,
         const bool batchMode = true,
         RangeSearchType rangeSearch = RangeSearchType(),
         PointSelectionPolicy pointSelector = PointSelectionPolicy());

  /**
   * Performs DBSCAN clustering on the data, returning number of clusters
   * and also the centroid of each cluster.
   *
   * @tparam MatType Only dense matrix type (e.g., arma::mat or coot::mat).
   * @param data Dataset to cluster.
   * @param centroids Matrix in which centroids are stored.
   */
  size_t Cluster(const MatType& data,
                 MatType& centroids);

  /**
   * Performs DBSCAN clustering on the data, returning number of clusters
   * and also the list of cluster assignments.  If assignments[i] == SIZE_MAX,
   * then the point is considered "noise".
   *
   * @tparam MatType Only dense matrix type (e.g., arma::mat or coot::mat).
   * @param data Dataset to cluster.
   * @param assignments Vector to store cluster assignments.
   */
  size_t Cluster(const MatType& data,
                 arma::Row<size_t>& assignments);

  /**
   * Performs DBSCAN clustering on the data, returning number of clusters,
   * the centroid of each cluster and also the list of cluster assignments.
   * If assignments[i] == SIZE_MAX, then the point is considered "noise".
   *
   * @tparam MatType Only dense matrix type (e.g., arma::mat or coot::mat).
   * @param data Dataset to cluster.
   * @param assignments Vector to store cluster assignments.
   * @param centroids Matrix in which centroids are stored.
   */
  size_t Cluster(const MatType& data,
                 arma::Row<size_t>& assignments,
                 MatType& centroids);

 private:
  //! Maximum distance between two points to be part of same cluster.
  ElemType epsilon;

  //! Minimum number of points to be in the epsilon-neighborhood (including
  //! itself) for the point to be a core-point.
  size_t minPoints;

  //! Whether or not to perform the search in batch mode.  If false, single
  bool batchMode;

  //! Instantiated range search policy.
  RangeSearchType rangeSearch;

  //! Instantiated point selection policy.
  PointSelectionPolicy pointSelector;

  /**
   * Performs DBSCAN clustering on the data, returning the number of clusters
   * and also the list of cluster assignments.  This searches each point
   * iteratively, and can save on RAM usage.  It may be slower than the batch
   * search with a dual-tree algorithm.
   *
   * @param data Dataset to cluster.
   * @param assignments Assignments for each point.
   * @param uf UnionFind structure that will be modified.
   */
  void PointwiseCluster(const MatType& data, UnionFind& uf);

  /**
   * Performs DBSCAN clustering on the data, returning number of clusters and
   * also the list of cluster assignments.  This can perform search in batch, so
   * it is well suited for dual-tree or naive search.
   *
   * @param data Dataset to cluster.
   * @param assignments Assignments for each point.
   * @param uf UnionFind structure that will be modified.
   */
  void BatchCluster(const MatType& data, UnionFind& uf);
};

} // namespace mlpack

// Include implementation.
#include "dbscan_impl.hpp"

#endif
