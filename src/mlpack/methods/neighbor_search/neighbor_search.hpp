/**
 * @file neighbor_search.hpp
 * @author Ryan Curtin
 *
 * Defines the NeighborSearch class, which performs an abstract
 * nearest-neighbor-like query on two datasets.
 *
 * This file is part of MLPACK 1.0.4.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP

#include <mlpack/core.hpp>
#include <vector>
#include <string>

#include <mlpack/core/tree/binary_space_tree.hpp>

#include <mlpack/core/metrics/lmetric.hpp>
#include "sort_policies/nearest_neighbor_sort.hpp"

namespace mlpack {
namespace neighbor /** Neighbor-search routines.  These include
                    * all-nearest-neighbors and all-furthest-neighbors
                    * searches. */ {

/**
 * Extra data for each node in the tree.  For neighbor searches, each node only
 * needs to store a bound on neighbor distances.
 */
template<typename SortPolicy>
class QueryStat
{
 private:
  //! The first bound on the node's neighbor distances (B_1).  This represents
  //! the worst candidate distance of any descendants of this node.
  double firstBound;
  //! The second bound on the node's neighbor distances (B_2).  This represents
  //! a bound on the worst distance of any descendants of this node assembled
  //! using the best descendant candidate distance modified by the furthest
  //! descendant distance.
  double secondBound;
  //! The better of the two bounds.
  double bound;

 public:
  /**
   * Initialize the statistic with the worst possible distance according to
   * our sorting policy.
   */
  QueryStat() :
      firstBound(SortPolicy::WorstDistance()),
      secondBound(SortPolicy::WorstDistance()),
      bound(SortPolicy::WorstDistance()) { }

  /**
   * Initialization for a fully initialized node.  In this case, we don't need
   * to worry about the node.
   */
  template<typename TreeType>
  QueryStat(TreeType& /* node */) :
      firstBound(SortPolicy::WorstDistance()),
      secondBound(SortPolicy::WorstDistance()),
      bound(SortPolicy::WorstDistance()) { }

  //! Get the first bound.
  double FirstBound() const { return firstBound; }
  //! Modify the first bound.
  double& FirstBound() { return firstBound; }
  //! Get the second bound.
  double SecondBound() const { return secondBound; }
  //! Modify the second bound.
  double& SecondBound() { return secondBound; }
  //! Get the overall bound (the better of the two bounds).
  double Bound() const { return bound; }
  //! Modify the overall bound (it should be the better of the two bounds).
  double& Bound() { return bound; }
};

/**
 * The NeighborSearch class is a template class for performing distance-based
 * neighbor searches.  It takes a query dataset and a reference dataset (or just
 * a reference dataset) and, for each point in the query dataset, finds the k
 * neighbors in the reference dataset which have the 'best' distance according
 * to a given sorting policy.  A constructor is given which takes only a
 * reference dataset, and if that constructor is used, the given reference
 * dataset is also used as the query dataset.
 *
 * The template parameters SortPolicy and Metric define the sort function used
 * and the metric (distance function) used.  More information on those classes
 * can be found in the NearestNeighborSort class and the kernel::ExampleKernel
 * class.
 *
 * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
 * @tparam MetricType The metric to use for computation.
 * @tparam TreeType The tree type to use.
 */
template<typename SortPolicy = NearestNeighborSort,
         typename MetricType = mlpack::metric::SquaredEuclideanDistance,
         typename TreeType = tree::BinarySpaceTree<bound::HRectBound<2>,
                                                   QueryStat<SortPolicy> > >
class NeighborSearch
{
 public:
  /**
   * Initialize the NeighborSearch object, passing both a query and reference
   * dataset.  Optionally, perform the computation in naive mode or single-tree
   * mode, and set the leaf size used for tree-building.  An initialized
   * distance metric can be given, for cases where the metric has internal data
   * (i.e. the distance::MahalanobisDistance class).
   *
   * This method will copy the matrices to internal copies, which are rearranged
   * during tree-building.  You can avoid this extra copy by pre-constructing
   * the trees and passing them using a diferent constructor.
   *
   * @param referenceSet Set of reference points.
   * @param querySet Set of query points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param leafSize Leaf size for tree construction (ignored if tree is given).
   * @param metric An optional instance of the MetricType class.
   */
  NeighborSearch(const typename TreeType::Mat& referenceSet,
                 const typename TreeType::Mat& querySet,
                 const bool naive = false,
                 const bool singleMode = false,
                 const size_t leafSize = 20,
                 const MetricType metric = MetricType());

  /**
   * Initialize the NeighborSearch object, passing only one dataset, which is
   * used as both the query and the reference dataset.  Optionally, perform the
   * computation in naive mode or single-tree mode, and set the leaf size used
   * for tree-building.  An initialized distance metric can be given, for cases
   * where the metric has internal data (i.e. the distance::MahalanobisDistance
   * class).
   *
   * If naive mode is being used and a pre-built tree is given, it may not work:
   * naive mode operates by building a one-node tree (the root node holds all
   * the points).  If that condition is not satisfied with the pre-built tree,
   * then naive mode will not work.
   *
   * @param referenceSet Set of reference points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param leafSize Leaf size for tree construction (ignored if tree is given).
   * @param metric An optional instance of the MetricType class.
   */
  NeighborSearch(const typename TreeType::Mat& referenceSet,
                 const bool naive = false,
                 const bool singleMode = false,
                 const size_t leafSize = 20,
                 const MetricType metric = MetricType());

  /**
   * Initialize the NeighborSearch object with the given datasets and
   * pre-constructed trees.  It is assumed that the points in referenceSet and
   * querySet correspond to the points in referenceTree and queryTree,
   * respectively.  Optionally, choose to use single-tree mode.  Naive mode is
   * not available as an option for this constructor; instead, to run naive
   * computation, construct a tree with all of the points in one leaf (i.e.
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
  NeighborSearch(TreeType* referenceTree,
                 TreeType* queryTree,
                 const typename TreeType::Mat& referenceSet,
                 const typename TreeType::Mat& querySet,
                 const bool singleMode = false,
                 const MetricType metric = MetricType());

  /**
   * Initialize the NeighborSearch object with the given reference dataset and
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
  NeighborSearch(TreeType* referenceTree,
                 const typename TreeType::Mat& referenceSet,
                 const bool singleMode = false,
                 const MetricType metric = MetricType());


  /**
   * Delete the NeighborSearch object. The tree is the only member we are
   * responsible for deleting.  The others will take care of themselves.
   */
  ~NeighborSearch();

  /**
   * Compute the nearest neighbors and store the output in the given matrices.
   * The matrices will be set to the size of n columns by k rows, where n is the
   * number of points in the query dataset and k is the number of neighbors
   * being searched for.
   *
   * @param k Number of neighbors to search for.
   * @param resultingNeighbors Matrix storing lists of neighbors for each query
   *     point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& resultingNeighbors,
              arma::mat& distances);

 private:
  //! Copy of reference dataset (if we need it, because tree building modifies
  //! it).
  arma::mat referenceCopy;
  //! Copy of query dataset (if we need it, because tree building modifies it).
  arma::mat queryCopy;

  //! Reference dataset.
  const arma::mat& referenceSet;
  //! Query dataset (may not be given).
  const arma::mat& querySet;

  //! Pointer to the root of the reference tree.
  TreeType* referenceTree;
  //! Pointer to the root of the query tree (might not exist).
  TreeType* queryTree;

  //! Indicates if we should free the reference tree at deletion time.
  bool ownReferenceTree;
  //! Indicates if we should free the query tree at deletion time.
  bool ownQueryTree;

  //! Indicates if O(n^2) naive search is being used.
  bool naive;
  //! Indicates if single-tree search is being used (opposed to dual-tree).
  bool singleMode;

  //! Instantiation of kernel.
  MetricType metric;

  //! Permutations of reference points during tree building.
  std::vector<size_t> oldFromNewReferences;
  //! Permutations of query points during tree building.
  std::vector<size_t> oldFromNewQueries;

  //! Total number of pruned nodes during the neighbor search.
  size_t numberOfPrunes;
}; // class NeighborSearch

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "neighbor_search_impl.hpp"

// Include convenience typedefs.
#include "typedef.hpp"

#endif
