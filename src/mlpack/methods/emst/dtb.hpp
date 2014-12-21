/**
 * @file dtb.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Contains an implementation of the DualTreeBoruvka algorithm for finding a
 * Euclidean Minimum Spanning Tree using the kd-tree data structure.
 *
 * @code
 * @inproceedings{
 *   author = {March, W.B., Ram, P., and Gray, A.G.},
 *   title = {{Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,
 *      Applications.}},
 *   booktitle = {Proceedings of the 16th ACM SIGKDD International Conference
 *      on Knowledge Discovery and Data Mining}
 *   series = {KDD 2010},
 *   year = {2010}
 * }
 * @endcode
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_EMST_DTB_HPP
#define __MLPACK_METHODS_EMST_DTB_HPP

#include "dtb_stat.hpp"
#include "edge_pair.hpp"

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <mlpack/core/tree/binary_space_tree.hpp>

namespace mlpack {
namespace emst /** Euclidean Minimum Spanning Trees. */ {

/**
 * Performs the MST calculation using the Dual-Tree Boruvka algorithm, using any
 * type of tree.
 *
 * For more information on the algorithm, see the following citation:
 *
 * @code
 * @inproceedings{
 *   author = {March, W.B., Ram, P., and Gray, A.G.},
 *   title = {{Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,
 *      Applications.}},
 *   booktitle = {Proceedings of the 16th ACM SIGKDD International Conference
 *      on Knowledge Discovery and Data Mining}
 *   series = {KDD 2010},
 *   year = {2010}
 * }
 * @endcode
 *
 * General usage of this class might be like this:
 *
 * @code
 * extern arma::mat data; // We want to find the MST of this dataset.
 * DualTreeBoruvka<> dtb(data); // Create the tree with default options.
 *
 * // Find the MST.
 * arma::mat mstResults;
 * dtb.ComputeMST(mstResults);
 * @endcode
 *
 * More advanced usage of the class can use different types of trees, pass in an
 * already-built tree, or compute the MST using the O(n^2) naive algorithm.
 *
 * @tparam MetricType The metric to use.  IMPORTANT: this hasn't really been
 * tested with anything other than the L2 metric, so user beware. Note that the
 * tree type needs to compute bounds using the same metric as the type
 * specified here.
 * @tparam TreeType Type of tree to use.  Should use DTBStat as a statistic.
 */
template<
  typename MetricType = metric::EuclideanDistance,
  typename TreeType = tree::BinarySpaceTree<bound::HRectBound<2>, DTBStat>
>
class DualTreeBoruvka
{
 private:
  //! Copy of the data (if necessary).
  typename TreeType::Mat dataCopy;
  //! Reference to the data (this is what should be used for accessing data).
  const typename TreeType::Mat& data;

  //! Pointer to the root of the tree.
  TreeType* tree;
  //! Indicates whether or not we "own" the tree.
  bool ownTree;

  //! Indicates whether or not O(n^2) naive mode will be used.
  bool naive;

  //! Edges.
  std::vector<EdgePair> edges; // We must use vector with non-numerical types.

  //! Connections.
  UnionFind connections;

  //! Permutations of points during tree building.
  std::vector<size_t> oldFromNew;
  //! List of edge nodes.
  arma::Col<size_t> neighborsInComponent;
  //! List of edge nodes.
  arma::Col<size_t> neighborsOutComponent;
  //! List of edge distances.
  arma::vec neighborsDistances;

  //! Total distance of the tree.
  double totalDist;

  //! The instantiated metric.
  MetricType metric;

  //! For sorting the edge list after the computation.
  struct SortEdgesHelper
  {
    bool operator()(const EdgePair& pairA, const EdgePair& pairB)
    {
      return (pairA.Distance() < pairB.Distance());
    }
  } SortFun;

 public:
  /**
   * Create the tree from the given dataset.  This copies the dataset to an
   * internal copy, because tree-building modifies the dataset.
   *
   * @param data Dataset to build a tree for.
   * @param naive Whether the computation should be done in O(n^2) naive mode.
   * @param leafSize The leaf size to be used during tree construction.
   */
  DualTreeBoruvka(const typename TreeType::Mat& dataset,
                  const bool naive = false,
                  const MetricType metric = MetricType());

  /**
   * Create the DualTreeBoruvka object with an already initialized tree.  This
   * will not copy the dataset, and can save a little processing power.  Naive
   * mode is not available as an option for this constructor; instead, to run
   * naive computation, construct a tree with all the points in one leaf (i.e.
   * leafSize = number of points).
   *
   * @note
   * Because tree-building (at least with BinarySpaceTree) modifies the ordering
   * of a matrix, be sure you pass the modified matrix to this object!  In
   * addition, mapping the points of the matrix back to their original indices
   * is not done when this constructor is used.
   * @endnote
   *
   * @param tree Pre-built tree.
   * @param dataset Dataset corresponding to the pre-built tree.
   */
  DualTreeBoruvka(TreeType* tree,
                  const typename TreeType::Mat& dataset,
                  const MetricType metric = MetricType());

  /**
   * Delete the tree, if it was created inside the object.
   */
  ~DualTreeBoruvka();

  /**
   * Iteratively find the nearest neighbor of each component until the MST is
   * complete.  The results will be a 3xN matrix (with N equal to the number of
   * edges in the minimum spanning tree).  The first row will contain the lesser
   * index of the edge; the second row will contain the greater index of the
   * edge; and the third row will contain the distance between the two edges.
   *
   * @param results Matrix which results will be stored in.
   */
  void ComputeMST(arma::mat& results);

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

 private:
  /**
   * Adds a single edge to the edge list
   */
  void AddEdge(const size_t e1, const size_t e2, const double distance);

  /**
   * Adds all the edges found in one iteration to the list of neighbors.
   */
  void AddAllEdges();

  /**
   * Unpermute the edge list and output it to results.
   */
  void EmitResults(arma::mat& results);

  /**
   * This function resets the values in the nodes of the tree nearest neighbor
   * distance, and checks for fully connected nodes.
   */
  void CleanupHelper(TreeType* tree);

  /**
   * The values stored in the tree must be reset on each iteration.
   */
  void Cleanup();

}; // class DualTreeBoruvka

}; // namespace emst
}; // namespace mlpack

#include "dtb_impl.hpp"

#endif // __MLPACK_METHODS_EMST_DTB_HPP
