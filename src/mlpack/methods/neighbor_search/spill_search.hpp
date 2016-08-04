/**
 * @file spill_search.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Defines the SpillSearch class, which performs a Hybrid sp-tree search on
 * two datasets.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_SPILL_SEARCH_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_SPILL_SEARCH_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include "sort_policies/nearest_neighbor_sort.hpp"
#include "neighbor_search.hpp"

namespace mlpack {
namespace neighbor {

// Forward declaration.
template<typename SortPolicy>
class TrainVisitor;

/**
 * The SpillSearch class is a template class for performing distance-based
 * neighbor searches with Spill Trees.  It takes a query dataset and a reference
 * dataset (or just a reference dataset) and, for each point in the query
 * dataset, finds the k neighbors in the reference dataset which have the 'best'
 * distance according to a given sorting policy.  A constructor is given which
 * takes only a reference dataset, and if that constructor is used, the given
 * reference dataset is also used as the query dataset.
 *
 * @tparam MetricType The metric to use for computation.
 * @tparam MatType The type of data matrix.
 * @tparam SplitType The class that partitions the dataset/points at a
 *     particular node into two parts. Its definition decides the way this split
 *     is done when building spill trees.
 */
template<typename MetricType = mlpack::metric::EuclideanDistance,
         typename MatType = arma::mat,
         template<typename HyperplaneMetricType>
             class HyperplaneType = tree::AxisOrthogonalHyperplane,
         template<typename SplitBoundT, typename SplitMatT> class SplitType =
             tree::MidpointSpaceSplit>
class SpillSearch
{
 public:
  //! Convenience typedef.
  typedef tree::SpillTree<MetricType, NeighborSearchStat<NearestNeighborSort>,
      MatType, HyperplaneType, SplitType> Tree;

  template<typename TreeMetricType,
           typename TreeStatType,
           typename TreeMatType>
  using TreeType = tree::SpillTree<TreeMetricType, TreeStatType, TreeMatType,
      HyperplaneType, SplitType>;

  /**
   * Initialize the SpillSearch object, passing a reference dataset (this is
   * the dataset which is searched).  Optionally, perform the computation in
   * naive mode or single-tree mode.  An initialized distance metric can be
   * given, for cases where the metric has internal data (i.e. the
   * distance::MahalanobisDistance class).
   *
   * @param referenceSet Set of reference points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param tau Overlapping size (non-negative).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric An optional instance of the MetricType class.
   */
  SpillSearch(const MatType& referenceSet,
              const bool naive = false,
              const bool singleMode = false,
              const double tau = 0,
              const double epsilon = 0,
              const MetricType metric = MetricType());

  /**
   * Initialize the SpillSearch object, taking ownership of the reference
   * dataset (this is the dataset which is searched).  Optionally, perform the
   * computation in naive mode or single-tree mode.  An initialized distance
   * metric can be given, for cases where the metric has internal data (i.e. the
   * distance::MahalanobisDistance class).
   *
   * @param referenceSet Set of reference points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param tau Overlapping size (non-negative).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric An optional instance of the MetricType class.
   */
  SpillSearch(MatType&& referenceSet,
              const bool naive = false,
              const bool singleMode = false,
              const double tau = 0,
              const double epsilon = 0,
              const MetricType metric = MetricType());

  /**
   * Initialize the SpillSearch object with the given pre-constructed
   * reference tree (this is the tree built on the points that will be
   * searched).  Optionally, choose to use single-tree mode.  Naive mode is not
   * available as an option for this constructor.  Additionally, an instantiated
   * distance metric can be given, for cases where the distance metric holds
   * data.
   *
   * @param referenceTree Pre-built tree for reference points.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param tau Overlapping size (non-negative).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric Instantiated distance metric.
   */
  SpillSearch(Tree* referenceTree,
              const bool singleMode = false,
              const double tau = 0,
              const double epsilon = 0,
              const MetricType metric = MetricType());

  /**
   * Create a SpillSearch object without any reference data.  If Search() is
   * called before a reference set is set with Train(), an exception will be
   * thrown.
   *
   * @param naive Whether to use naive search.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param tau Overlapping size (non-negative).
   * @param epsilon Relative approximate error (non-negative).
   * @param metric Instantiated metric.
   */
  SpillSearch(const bool naive = false,
              const bool singleMode = false,
              const double tau = 0,
              const double epsilon = 0,
              const MetricType metric = MetricType());


  /**
   * Delete the SpillSearch object. The tree is the only member we are
   * responsible for deleting.  The others will take care of themselves.
   */
  ~SpillSearch();

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
   *
   * @param referenceTree Pre-built tree for reference points.
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
   */
  void Search(Tree* queryTree,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

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

  //! Return the total number of base case evaluations performed during the last
  //! search.
  size_t BaseCases() const { return neighborSearch.BaseCases(); }

  //! Return the number of node combination scores during the last search.
  size_t Scores() const { return neighborSearch.Scores(); }

  //! Access whether or not search is done in naive linear scan mode.
  bool Naive() const { return neighborSearch.Naive(); }
  //! Modify whether or not search is done in naive linear scan mode.
  bool& Naive() { return neighborSearch.Naive(); }

  //! Access whether or not search is done in single-tree mode.
  bool SingleMode() const { return neighborSearch.SingleMode(); }
  //! Modify whether or not search is done in single-tree mode.
  bool& SingleMode() { return neighborSearch.SingleMode(); }

  //! Access the relative error to be considered in approximate search.
  double Epsilon() const { return neighborSearch.Epsilon(); }
  //! Modify the relative error to be considered in approximate search.
  double& Epsilon() { return neighborSearch.Epsilon(); }

  //! Access the overlapping size.
  double Tau() const { return tau; }

  //! Access the reference dataset.
  const MatType& ReferenceSet() const { return neighborSearch.ReferenceSet(); }

  //! Serialize the SpillSearch model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Internal instance of NeighborSearch class.
  NeighborSearch<NearestNeighborSort, MetricType, MatType, TreeType>
      neighborSearch;

  //! Overlapping size.
  double tau;

  //! The NSModel class should have access to internal members.
  template<typename SortPolicy>
  friend class TrainVisitor;
}; // class SpillSearch

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "spill_search_impl.hpp"

#endif
