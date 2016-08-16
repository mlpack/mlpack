/**
 * @file neighbor_search.hpp
 * @author Ryan Curtin
 *
 * Defines the NeighborSearch class, which performs an abstract
 * nearest-neighbor-like query on two datasets.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP

#include <mlpack/core.hpp>
#include <vector>
#include <string>

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
#include <mlpack/core/tree/binary_space_tree/binary_space_tree.hpp>

#include <mlpack/core/metrics/lmetric.hpp>
#include "neighbor_search_stat.hpp"
#include "sort_policies/nearest_neighbor_sort.hpp"
#include "neighbor_search_rules.hpp"

namespace mlpack {
namespace neighbor /** Neighbor-search routines.  These include
                    * all-nearest-neighbors and all-furthest-neighbors
                    * searches. */ {

// Forward declaration.
template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
class SpillSearch;

// Forward declaration.
template<typename SortPolicy>
class TrainVisitor;

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
 * @tparam MatType The type of data matrix.
 * @tparam TreeType The tree type to use; must adhere to the TreeType API.
 * @tparam TraversalType The type of traversal to use (defaults to the tree's
 *      default traverser).
 */
template<typename SortPolicy = NearestNeighborSort,
         typename MetricType = mlpack::metric::EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = tree::KDTree,
         template<typename RuleType> class TraversalType =
             TreeType<MetricType,
                      NeighborSearchStat<SortPolicy>,
                      MatType>::template DualTreeTraverser>
class NeighborSearch
{
 public:
  //! Convenience typedef.
  typedef TreeType<MetricType, NeighborSearchStat<SortPolicy>, MatType> Tree;

  /**
   * Initialize the NeighborSearch object, passing a reference dataset (this is
   * the dataset which is searched).  Optionally, perform the computation in
   * naive mode or single-tree mode.  An initialized distance metric can be
   * given, for cases where the metric has internal data (i.e. the
   * distance::MahalanobisDistance class).
   *
   * This method will copy the matrices to internal copies, which are rearranged
   * during tree-building.  You can avoid this extra copy by pre-constructing
   * the trees and passing them using a different constructor, or by using the
   * construct that takes an rvalue reference to the dataset.
   *
   * @param referenceSet Set of reference points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric An optional instance of the MetricType class.
   */
  NeighborSearch(const MatType& referenceSet,
                 const bool naive = false,
                 const bool singleMode = false,
                 const double epsilon = 0,
                 const MetricType metric = MetricType());

  /**
   * Initialize the NeighborSearch object, taking ownership of the reference
   * dataset (this is the dataset which is searched).  Optionally, perform the
   * computation in naive mode or single-tree mode.  An initialized distance
   * metric can be given, for cases where the metric has internal data (i.e. the
   * distance::MahalanobisDistance class).
   *
   * This method will not copy the data matrix, but will take ownership of it,
   * and depending on the type of tree used, may rearrange the points.  If you
   * would rather a copy be made, consider using the constructor that takes a
   * const reference to the data instead.
   *
   * @param referenceSet Set of reference points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric An optional instance of the MetricType class.
   */
  NeighborSearch(MatType&& referenceSet,
                 const bool naive = false,
                 const bool singleMode = false,
                 const double epsilon = 0,
                 const MetricType metric = MetricType());

  /**
   * Initialize the NeighborSearch object with the given pre-constructed
   * reference tree (this is the tree built on the points that will be
   * searched).  Optionally, choose to use single-tree mode.  Naive mode is not
   * available as an option for this constructor.  Additionally, an instantiated
   * distance metric can be given, for cases where the distance metric holds
   * data.
   *
   * There is no copying of the data matrices in this constructor (because
   * tree-building is not necessary), so this is the constructor to use when
   * copies absolutely must be avoided.
   *
   * @note
   * Mapping the points of the matrix back to their original indices is not done
   * when this constructor is used, so if the tree type you are using maps
   * points (like BinarySpaceTree), then you will have to perform the re-mapping
   * manually.
   * @endnote
   *
   * @param referenceTree Pre-built tree for reference points.
   * @param referenceSet Set of reference points corresponding to referenceTree.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric Instantiated distance metric.
   */
  NeighborSearch(Tree* referenceTree,
                 const bool singleMode = false,
                 const double epsilon = 0,
                 const MetricType metric = MetricType());

  /**
   * Create a NeighborSearch object without any reference data.  If Search() is
   * called before a reference set is set with Train(), an exception will be
   * thrown.
   *
   * @param naive Whether to use naive search.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric Instantiated metric.
   */
  NeighborSearch(const bool naive = false,
                 const bool singleMode = false,
                 const double epsilon = 0,
                 const MetricType metric = MetricType());


  /**
   * Delete the NeighborSearch object. The tree is the only member we are
   * responsible for deleting.  The others will take care of themselves.
   */
  ~NeighborSearch();

  /**
   * Set the reference set to a new reference set, and build a tree if
   * necessary.  This method is called 'Train()' in order to match the rest of
   * the mlpack abstractions, even though calling this "training" is maybe a bit
   * of a stretch.
   *
   * @param referenceSet New set of reference data.
   */
  void Train(const MatType& referenceSet);

  /**
   * Set the reference set to a new reference set, taking ownership of the set,
   * and build a tree if necessary.  This method is called 'Train()' in order to
   * match the rest of the mlpack abstractions, even though calling this
   * "training" is maybe a bit of a stretch.
   *
   * @param referenceSet New set of reference data.
   */
  void Train(MatType&& referenceSet);

  /**
   * Set the reference tree to a new reference tree.
   */
  void Train(Tree* referenceTree);

  /**
   * For each point in the query set, compute the nearest neighbors and store
   * the output in the given matrices.  The matrices will be set to the size of
   * n columns by k rows, where n is the number of points in the query dataset
   * and k is the number of neighbors being searched for.
   *
   * If querySet contains only a few query points, the extra cost of building a
   * tree on the points for dual-tree search may not be warranted, and it may be
   * worthwhile to set singleMode = false (either in the constructor or with
   * SingleMode()).
   *
   * @param querySet Set of query points (can be just one point).
   * @param k Number of neighbors to search for.
   * @param neighbors Matrix storing lists of neighbors for each query point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void Search(const MatType& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * Given a pre-built query tree, search for the nearest neighbors of each
   * point in the query tree, storing the output in the given matrices.  The
   * matrices will be set to the size of n columns by k rows, where n is the
   * number of points in the query dataset and k is the number of neighbors
   * being searched for.
   *
   * Note that if you are calling Search() multiple times with a single query
   * tree, you need to reset the bounds in the statistic of each query node,
   * otherwise the result may be wrong!  You can do this by calling
   * TreeType::Stat()::Reset() on each node in the query tree.
   *
   * @param queryTree Tree built on query points.
   * @param k Number of neighbors to search for.
   * @param neighbors Matrix storing lists of neighbors for each query point.
   * @param distances Matrix storing distances of neighbors for each query
   *      point.
   * @param sameSet Denotes whether or not the reference and query sets are the
   *      same.
   */
  void Search(Tree* queryTree,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances,
              bool sameSet = false);

  /**
   * Search for the nearest neighbors of every point in the reference set.  This
   * is basically equivalent to calling any other overload of Search() with the
   * reference set as the query set; so, this lets you do
   * all-k-nearest-neighbors search.  The results are stored in the given
   * matrices.  The matrices will be set to the size of n columns by k rows,
   * where n is the number of points in the query dataset and k is the number of
   * neighbors being searched for.
   *
   * @param k Number of neighbors to search for.
   * @param neighbors Matrix storing lists of neighbors for each query point.
   * @param distances Matrix storing distances of neighbors for each query
   *      point.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * Calculate the average relative error (effective error) between the
   * distances calculated and the true distances provided.  The input matrices
   * must have the same size.
   *
   * Cases where the true distance is zero (the same point) or the calculated
   * distance is SortPolicy::WorstDistance() (didn't find enough points) will be
   * ignored.
   *
   * @param foundDistances Matrix storing lists of calculated distances for each
   *     query point.
   * @param realDistances Matrix storing lists of true best distances for each
   *     query point.
   * @return Average relative error.
   */
  static double EffectiveError(arma::mat& foundDistances,
                               arma::mat& realDistances);

  /**
   * Calculate the recall (% of neighbors found) given the list of found
   * neighbors and the true set of neighbors.  The recall returned will be in
   * the range [0, 1].
   *
   * @param foundNeighbors Matrix storing lists of calculated neighbors for each
   *     query point.
   * @param realNeighbors Matrix storing lists of true best neighbors for each
   *     query point.
   * @return Recall.
   */
  static double Recall(arma::Mat<size_t>& foundNeighbors,
                       arma::Mat<size_t>& realNeighbors);

  //! Return the total number of base case evaluations performed during the last
  //! search.
  size_t BaseCases() const { return baseCases; }

  //! Return the number of node combination scores during the last search.
  size_t Scores() const { return scores; }

  //! Access whether or not search is done in naive linear scan mode.
  bool Naive() const { return naive; }
  //! Modify whether or not search is done in naive linear scan mode.
  bool& Naive() { return naive; }

  //! Access whether or not search is done in single-tree mode.
  bool SingleMode() const { return singleMode; }
  //! Modify whether or not search is done in single-tree mode.
  bool& SingleMode() { return singleMode; }

  //! Access the relative error to be considered in approximate search.
  double Epsilon() const { return epsilon; }
  //! Modify the relative error to be considered in approximate search.
  double& Epsilon() { return epsilon; }

  //! Access the reference dataset.
  const MatType& ReferenceSet() const { return *referenceSet; }

  //! Serialize the NeighborSearch model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Permutations of reference points during tree building.
  std::vector<size_t> oldFromNewReferences;
  //! Pointer to the root of the reference tree.
  Tree* referenceTree;
  //! Reference dataset.  In some situations we may be the owner of this.
  const MatType* referenceSet;

  //! If true, this object created the trees and is responsible for them.
  bool treeOwner;
  //! If true, we own the reference set.
  bool setOwner;

  //! Indicates if O(n^2) naive search is being used.
  bool naive;
  //! Indicates if single-tree search is being used (as opposed to dual-tree).
  bool singleMode;
  //! Indicates the relative error to be considered in approximate search.
  double epsilon;

  //! Instantiation of metric.
  MetricType metric;

  //! The total number of base cases.
  size_t baseCases;
  //! The total number of scores (applicable for non-naive search).
  size_t scores;

  //! If this is true, the reference tree bounds need to be reset on a call to
  //! Search() without a query set.
  bool treeNeedsReset;

  //! The NSModel class should have access to internal members.
  template<typename SortPol>
  friend class TrainVisitor;

  template<typename MetricT,
           typename MatT,
           template<typename HyperplaneMetricType> class HyperplaneType,
           template<typename SplitBoundT, typename SplitMatT> class SplitType>
  friend class SpillSearch;
}; // class NeighborSearch

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "neighbor_search_impl.hpp"

// Include convenience typedefs.
#include "typedef.hpp"

#endif
