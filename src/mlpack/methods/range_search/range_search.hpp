/**
 * @file range_search.hpp
 * @author Ryan Curtin
 *
 * Defines the RangeSearch class, which performs a generalized range search on
 * points.
 * This file is part of MLPACK 1.0.2.
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
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/metrics/lmetric.hpp>

#include <mlpack/core/tree/binary_space_tree.hpp>

namespace mlpack {
namespace range /** Range-search routines. */ {

/**
 * The RangeSearch class is a template class for performing range searches.
 */
template<typename MetricType = mlpack::metric::SquaredEuclideanDistance,
         typename TreeType = tree::BinarySpaceTree<bound::HRectBound<2>,
                                                   tree::EmptyStatistic> >
class RangeSearch
{
 public:
  /**
   * Initialize the RangeSearch object with a different reference set and a
   * query set.  Optionally, perform the computation in naive mode or
   * single-tree mode, and set the leaf size used for tree-building.
   * Additionally, an instantiated metric can be given, for cases where the
   * distance metric holds data.
   *
   * This method will copy the matrices to internal copies, which are rearranged
   * during tree-building.  You can avoid this extra copy by pre-constructing
   * the trees and passing them using a different constructor.
   *
   * @param referenceSet Reference dataset.
   * @param querySet Query dataset.
   * @param naive Whether the computation should be done in O(n^2) naive mode.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param leafSize The leaf size to be used during tree construction.
   * @param metric Instantiated distance metric.
   */
  RangeSearch(const typename TreeType::Mat& referenceSet,
              const typename TreeType::Mat& querySet,
              const bool naive = false,
              const bool singleMode = false,
              const size_t leafSize = 20,
              const MetricType metric = MetricType());

  /**
   * Initialize the RangeSearch object with only a reference set, which will
   * also be used as a query set.  Optionally, perform the computation in naive
   * mode or single-tree mode, and set the leaf size used for tree-building.
   * Additionally an instantiated metric can be given, for cases where the
   * distance metric holds data.
   *
   * This method will copy the reference matrix to an internal copy, which is
   * rearranged during tree-building.  You can avoid this extra copy by
   * pre-constructing the reference tree and passing it using a different
   * constructor.
   *
   * @param referenceSet Reference dataset.
   * @param naive Whether the computation should be done in O(n^2) naive mode.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param leafSize The leaf size to be used during tree construction.
   * @param metric Instantiated distance metric.
   */
  RangeSearch(const typename TreeType::Mat& referenceSet,
              const bool naive = false,
              const bool singleMode = false,
              const size_t leafSize = 20,
              const MetricType metric = MetricType());

  /**
   * Initialize the RangeSearch object with the given datasets and
   * pre-constructed trees.  It is assumed that the points in referenceSet and
   * querySet correspond to the points in referenceTree and queryTree,
   * respectively.  Optionally, choose to use single-tree mode.  Naive
   * mode is not available as an option for this constructor; instead, to run
   * naive computation, construct a tree with all the points in one leaf (i.e.
   * leafSize = number of points).  Additionally, an instantiated distance
   * metric can be given, for cases where the distance metric holds data.
   *
   * There is no copying of the data matrices in this constructor (because
   * tree-building is not necessary), so this is the constructor to use when
   * copies absolutely must be avoided.
   *
   * @note
   * Because tree-building (at least with BinarySpaceTree) modifies the ordering
   * of a matrix, be sure you pass the modified matrix to this object!  In
   * addition, mapping the points of the matrix back to their original indices
   * is not done when this constructor is used.
   * @endnote
   *
   * @param referenceTree Pre-built tree for reference points.
   * @param queryTree Pre-built tree for query points.
   * @param referenceSet Set of reference points corresponding to referenceTree.
   * @param querySet Set of query points corresponding to queryTree.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param metric Instantiated distance metric.
   */
  RangeSearch(TreeType* referenceTree,
              TreeType* queryTree,
              const typename TreeType::Mat& referenceSet,
              const typename TreeType::Mat& querySet,
              const bool singleMode = false,
              const MetricType metric = MetricType());

  /**
   * Initialize the RangeSearch object with the given reference dataset and
   * pre-constructed tree.  It is assumed that the points in referenceSet
   * correspond to the points in referenceTree.  Optionally, choose to use
   * single-tree mode.  Naive mode is not available as an option for this
   * constructor; instead, to run naive computation, construct a tree with all
   * the points in one leaf (i.e. leafSize = number of points).  Additionally,
   * an instantiated distance metric can be given, for the case where the
   * distance metric holds data.
   *
   * There is no copying of the data matrices in this constructor (because
   * tree-building is not necessary), so this is the constructor to use when
   * copies absolutely must be avoided.
   *
   * @note
   * Because tree-building (at least with BinarySpaceTree) modifies the ordering
   * of a matrix, be sure you pass the modified matrix to this object!  In
   * addition, mapping the points of the matrix back to their original indices
   * is not done when this constructor is used.
   * @endnote
   *
   * @param referenceTree Pre-built tree for reference points.
   * @param referenceSet Set of reference points corresponding to referenceTree.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param metric Instantiated distance metric.
   */
  RangeSearch(TreeType* referenceTree,
              const typename TreeType::Mat& referenceSet,
              const bool singleMode = false,
              const MetricType metric = MetricType());

  /**
   * Destroy the RangeSearch object.  If trees were created, they will be
   * deleted.
   */
  ~RangeSearch();

  /**
   * Search for all points in the given range, returning the results in the
   * neighbors and distances objects.  Each entry in the external vector
   * corresponds to a query point.  Each of these entries holds a vector which
   * contains the indices and distances of the reference points falling into the
   * given range.
   *
   * That is:
   *
   * - neighbors.size() and distances.size() both equal the number of query
   *   points.
   *
   * - neighbors[i] contains the indices of all the points in the reference set
   *   which have distances inside the given range to query point i.
   *
   * - distances[i] contains all of the distances corresponding to the indices
   *   contained in neighbors[i].
   *
   * - neighbors[i] and distances[i] are not sorted in any particular order.
   *
   * @param range Range of distances in which to search.
   * @param neighbors Object which will hold the list of neighbors for each
   *      point which fell into the given range, for each query point.
   * @param distances Object which will hold the list of distances for each
   *      point which fell into the given range, for each query point.
   */
  void Search(const math::Range& range,
              std::vector<std::vector<size_t> >& neighbors,
              std::vector<std::vector<double> >& distances);

 private:
  /**
   * Compute the base case, when both referenceNode and queryNode are leaves
   * containing points.
   *
   * @param referenceNode Reference node (must be a leaf).
   * @param queryNode Query node (must be a leaf).
   * @param range Range of distances to search for.
   * @param neighbors Object holding list of neighbors.
   * @param distances Object holding list of distances.
   */
  void ComputeBaseCase(const TreeType* referenceNode,
                       const TreeType* queryNode,
                       const math::Range& range,
                       std::vector<std::vector<size_t> >& neighbors,
                       std::vector<std::vector<double> >& distances) const;

  /**
   * Perform the dual-tree recursion, which will recurse until the base case is
   * necessary.
   *
   * @param referenceNode Reference node.
   * @param queryNode Query node.
   * @param range Range of distances to search for.
   * @param neighbors Object holding list of neighbors.
   * @param distances Object holding list of distances.
   */
  void DualTreeRecursion(const TreeType* referenceNode,
                         const TreeType* queryNode,
                         const math::Range& range,
                         std::vector<std::vector<size_t> >& neighbors,
                         std::vector<std::vector<double> >& distances);

  /**
   * Perform the single-tree recursion, which will recurse down the reference
   * tree to get the results for a single point.
   *
   * @param referenceNode Reference node.
   * @param queryPoint Point to query for.
   * @param queryIndex Index of query node.
   * @param range Range of distances to search for.
   * @param neighbors Object holding list of neighbors.
   * @param distances Object holding list of distances.
   */
  template<typename VecType>
  void SingleTreeRecursion(const TreeType* referenceNode,
                           const VecType& queryPoint,
                           const size_t queryIndex,
                           const math::Range& range,
                           std::vector<size_t>& neighbors,
                           std::vector<double>& distances);

  //! Copy of reference matrix; used when a tree is built internally.
  typename TreeType::Mat referenceCopy;
  //! Copy of query matrix; used when a tree is built internally.
  typename TreeType::Mat queryCopy;

  //! Reference set (data should be accessed using this).
  const typename TreeType::Mat& referenceSet;
  //! Query set (data should be accessed using this).
  const typename TreeType::Mat& querySet;

  //! Reference tree.
  TreeType* referenceTree;
  //! Query tree (may be NULL).
  TreeType* queryTree;

  //! Mappings to old reference indices (used when this object builds trees).
  std::vector<size_t> oldFromNewReferences;
  //! Mappings to old query indices (used when this object builds trees).
  std::vector<size_t> oldFromNewQueries;

  //! Indicates ownership of the reference tree (meaning we need to delete it).
  bool ownReferenceTree;
  //! Indicates ownership of the query tree (meaning we need to delete it).
  bool ownQueryTree;

  //! If true, O(n^2) naive computation is used.
  bool naive;
  //! If true, single-tree computation is used.
  bool singleMode;

  //! Instantiated distance metric.
  MetricType metric;

  //! The number of pruned nodes during computation.
  size_t numberOfPrunes;
};

}; // namespace range
}; // namespace mlpack

// Include implementation.
#include "range_search_impl.hpp"

#endif
