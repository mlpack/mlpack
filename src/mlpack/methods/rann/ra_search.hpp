/**
 * @file ra_search.hpp
 * @author Parikshit Ram
 *
 * Defines the RASearch class, which performs an abstract rank-approximate
 * nearest/farthest neighbor query on two datasets.
 *
 * The details of this method can be found in the following paper:
 *
 * @inproceedings{ram2009rank,
 *   title={{Rank-Approximate Nearest Neighbor Search: Retaining Meaning and
 *       Speed in High Dimensions}},
 *   author={{Ram, P. and Lee, D. and Ouyang, H. and Gray, A. G.}},
 *   booktitle={{Advances of Neural Information Processing Systems}},
 *   year={2009}
 * }
 *
 * This file is part of MLPACK 1.0.10.
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
#ifndef __MLPACK_METHODS_RANN_RA_SEARCH_HPP
#define __MLPACK_METHODS_RANN_RA_SEARCH_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/tree/binary_space_tree.hpp>

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>

#include "ra_query_stat.hpp"

/**
 * The RASearch class: This class provides a generic manner to perform
 * rank-approximate search via random-sampling. If the 'naive' option is chosen,
 * this rank-approximate search will be done by randomly sampled from the whole
 * set. If the 'naive' option is not chosen, the sampling is done in a
 * stratified manner in the tree as mentioned in the algorithms in Figure 2 of
 * the following paper:
 *
 * @inproceedings{ram2009rank,
 *   title={{Rank-Approximate Nearest Neighbor Search: Retaining Meaning and
 *       Speed in High Dimensions}},
 *   author={{Ram, P. and Lee, D. and Ouyang, H. and Gray, A. G.}},
 *   booktitle={{Advances of Neural Information Processing Systems}},
 *   year={2009}
 * }
 *
 * RASearch is currently known to not work with ball trees (#356).
 *
 * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
 * @tparam MetricType The metric to use for computation.
 * @tparam TreeType The tree type to use.
 */
template<typename SortPolicy = NearestNeighborSort,
         typename MetricType = mlpack::metric::SquaredEuclideanDistance,
         typename TreeType = tree::BinarySpaceTree<bound::HRectBound<2, false>,
                                                   RAQueryStat<SortPolicy> > >
class RASearch
{
 public:
  /**
   * Initialize the RASearch object, passing both a query and reference dataset.
   * Optionally, perform the computation in naive mode or single-tree mode, and
   * set the leaf size used for tree-building.  An initialized distance metric
   * can be given, for cases where the metric has internal data (i.e. the
   * distance::MahalanobisDistance class).
   *
   * This method will copy the matrices to internal copies, which are rearranged
   * during tree-building.  You can avoid this extra copy by pre-constructing
   * the trees and passing them using a diferent constructor.
   *
   * @param referenceSet Set of reference points.
   * @param querySet Set of query points.
   * @param naive If true, the rank-approximate search will be performed by
   *      directly sampling the whole set instead of using the stratified
   *      sampling on the tree.
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param leafSize Leaf size for tree construction (ignored if tree is given).
   * @param metric An optional instance of the MetricType class.
   */
  RASearch(const typename TreeType::Mat& referenceSet,
           const typename TreeType::Mat& querySet,
           const bool naive = false,
           const bool singleMode = false,
           const MetricType metric = MetricType());

  /**
   * Initialize the RASearch object, passing only one dataset, which is
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
   * @param naive If true, the rank-approximate search will be performed
   *      by directly sampling the whole set instead of using the stratified
   *      sampling on the tree.
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param leafSize Leaf size for tree construction (ignored if tree is given).
   * @param metric An optional instance of the MetricType class.
   */
  RASearch(const typename TreeType::Mat& referenceSet,
           const bool naive = false,
           const bool singleMode = false,
           const MetricType metric = MetricType());

  /**
   * Initialize the RASearch object with the given datasets and
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
  RASearch(TreeType* referenceTree,
           TreeType* queryTree,
           const typename TreeType::Mat& referenceSet,
           const typename TreeType::Mat& querySet,
           const bool singleMode = false,
           const MetricType metric = MetricType());

  /**
   * Initialize the RASearch object with the given reference dataset and
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
  RASearch(TreeType* referenceTree,
           const typename TreeType::Mat& referenceSet,
           const bool singleMode = false,
           const MetricType metric = MetricType());

  /**
   * Delete the RASearch object. The tree is the only member we are
   * responsible for deleting.  The others will take care of themselves.
   */
  ~RASearch();

  /**
   * Compute the rank approximate nearest neighbors and store the output in the
   * given matrices. The matrices will be set to the size of n columns by k
   * rows, where n is the number of points in the query dataset and k is the
   * number of neighbors being searched for.
   *
   * Note that tau, the rank-approximation parameter, specifies that we are
   * looking for k neighbors with probability alpha of being in the top tau
   * percent of nearest neighbors.  So, as an example, if our dataset has 1000
   * points, and we want 5 nearest neighbors with 95% probability of being in
   * the top 5% of nearest neighbors (or, the top 50 nearest neighbors), we set
   * k = 5, tau = 5, and alpha = 0.95.
   *
   * The method will fail (and issue a failure message) if the value of tau is
   * too low: tau must be set such that the number of points in the
   * corresponding percentile of the data is greater than k.  Thus, if we choose
   * tau = 0.1 with a dataset of 1000 points and k = 5, then we are attempting
   * to choose 5 nearest neighbors out of the closest 1 point -- this is
   * invalid.
   *
   * @param k Number of neighbors to search for.
   * @param resultingNeighbors Matrix storing lists of neighbors for each query
   *     point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   * @param tau The rank-approximation in percentile of the data. The default
   *     value is 5%.
   * @param alpha The desired success probability. The default value is 0.95.
   * @param sampleAtLeaves Sample at leaves for faster but less accurate
   *      computation. This defaults to 'false'.
   * @param firstLeafExact Traverse to the first leaf without approximation.
   *     This can ensure that the query definitely finds its (near) duplicate
   *     if there exists one.  This defaults to 'false' for now.
   * @param singleSampleLimit The limit on the largest node that can be
   *     approximated by sampling. This defaults to 20.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& resultingNeighbors,
              arma::mat& distances,
              const double tau = 5,
              const double alpha = 0.95,
              const bool sampleAtLeaves = false,
              const bool firstLeafExact = false,
              const size_t singleSampleLimit = 20);

  /**
   * This function recursively resets the RAQueryStat of the queryTree to set
   * 'bound' to WorstDistance and the 'numSamplesMade' to 0. This allows a user
   * to perform multiple searches on the same pair of trees, possibly with
   * different levels of approximation without requiring to build a new pair of
   * trees for every new (approximate) search.
   */
  void ResetQueryTree();

  // Returns a string representation of this object.
  std::string ToString() const;

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

  //! If true, this object created the trees and is responsible for them.
  bool treeOwner;
  //! Indicates if a separate query set was passed.
  bool hasQuerySet;

  //! Indicates if naive random sampling on the set is being used.
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

  /**
   * @param treeNode The node of the tree whose RAQueryStat is reset
   *     and whose children are to be explored recursively.
   */
  void ResetRAQueryStat(TreeType* treeNode);
}; // class RASearch

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "ra_search_impl.hpp"

// Include convenient typedefs.
#include "ra_typedef.hpp"

#endif
