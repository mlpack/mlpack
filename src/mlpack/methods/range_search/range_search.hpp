/**
 * @file range_search.hpp
 * @author Ryan Curtin
 *
 * Defines the RangeSearch class, which performs a generalized range search on
 * points.
 */
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include "range_search_stat.hpp"

namespace mlpack {
namespace range /** Range-search routines. */ {

/**
 * The RangeSearch class is a template class for performing range searches.  It
 * is implemented in the style of a generalized tree-independent dual-tree
 * algorithm; for more details on the actual algorithm, see the RangeSearchRules
 * class.
 *
 * @tparam MetricType Metric to use for range search calculations.
 * @tparam MatType Type of data to use.
 * @tparam TreeType Type of tree to use; must satisfy the TreeType policy API.
 */
template<typename MetricType = metric::EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType = tree::KDTree>
class RangeSearch
{
 public:
  //! Convenience typedef.
  typedef TreeType<MetricType, RangeSearchStat, MatType> Tree;

  /**
   * Initialize the RangeSearch object with a given reference dataset (this is
   * the dataset which is searched).  Optionally, perform the computation in
   * naive mode or single-tree mode. Additionally, an instantiated metric can be
   * given, for cases where the distance metric holds data.
   *
   * This method will copy the matrices to internal copies, which are rearranged
   * during tree-building.  You can avoid this extra copy by pre-constructing
   * the trees and passing them using a different constructor.
   *
   * @param referenceSet Reference dataset.
   * @param naive Whether the computation should be done in O(n^2) naive mode.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param metric Instantiated distance metric.
   */
  RangeSearch(const MatType& referenceSet,
              const bool naive = false,
              const bool singleMode = false,
              const MetricType metric = MetricType());

  /**
   * Initialize the RangeSearch object with the given pre-constructed reference
   * tree (this is the tree built on the reference set, which is the set that is
   * searched).  Optionally, choose to use single-tree mode, which will not
   * build a tree on query points.  Naive mode is not available as an option for
   * this constructor.  Additionally, an instantiated distance metric can be
   * given, for cases where the distance metric holds data.
   *
   * There is no copying of the data matrices in this constructor (because
   * tree-building is not necessary), so this is the constructor to use when
   * copies absolutely must be avoided.
   *
   * @note
   * Because tree-building (at least with BinarySpaceTree) modifies the ordering
   * of a matrix, be aware that mapping of the points back to their original
   * indices is not done when this constructor is used.
   * @endnote
   *
   * @param referenceTree Pre-built tree for reference points.
   * @param referenceSet Set of reference points corresponding to referenceTree.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param metric Instantiated distance metric.
   */
  RangeSearch(Tree* referenceTree,
              const bool singleMode = false,
              const MetricType metric = MetricType());

  /**
   * Destroy the RangeSearch object.  If trees were created, they will be
   * deleted.
   */
  ~RangeSearch();

  /**
   * Search for all reference points in the given range for each point in the
   * query set, returning the results in the neighbors and distances objects.
   * Each entry in the external vector corresponds to a query point.  Each of
   * these entries holds a vector which contains the indices and distances of
   * the reference points falling into the given range.
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
   * @param querySet Set of query points to search with.
   * @param range Range of distances in which to search.
   * @param neighbors Object which will hold the list of neighbors for each
   *      point which fell into the given range, for each query point.
   * @param distances Object which will hold the list of distances for each
   *      point which fell into the given range, for each query point.
   */
  void Search(const MatType& querySet,
              const math::Range& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<double>>& distances);

  /**
   * Given a pre-built query tree, search for all reference points in the given
   * range for each point in the query set, returning the results in the
   * neighbors and distances objects.
   *
   * Each entry in the external vector corresponds to a query point.  Each of
   * these entries holds a vector which contains the indices and distances of
   * the reference points falling into the given range.
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
   * If either naive or singleMode are set to true, this will throw an
   * invalid_argument exception; passing in a query tree implies dual-tree
   * search.
   *
   * If you want to use the reference tree as the query tree, instead call the
   * overload of Search() that does not take a query set.
   *
   * @param queryTree Tree built on query points.
   * @param range Range of distances in which to search.
   * @param neighbors Object which will hold the list of neighbors for each
   *      point which fell into the given range, for each query point.
   * @param distances Object which will hold the list of distances for each
   *      point which fell into the given range, for each query point.
   */
  void Search(Tree* queryTree,
              const math::Range& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<double>>& distances);

  /**
   * Search for all points in the given range for each point in the reference
   * set (which was passed to the constructor), returning the results in the
   * neighbors and distances objects.  This means that the query set and the
   * reference set are the same.
   *
   * Each entry in the external vector corresponds to a query point.  Each of
   * these entries holds a vector which contains the indices and distances of
   * the reference points falling into the given range.
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
   * @param queryTree Tree built on query points.
   * @param range Range of distances in which to search.
   * @param neighbors Object which will hold the list of neighbors for each
   *      point which fell into the given range, for each query point.
   * @param distances Object which will hold the list of distances for each
   *      point which fell into the given range, for each query point.
   */
  void Search(const math::Range& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<double>>& distances);

  //! Returns a string representation of this object.
  std::string ToString() const;

  //! Return the reference tree (or NULL if in naive mode).
  Tree* ReferenceTree() { return referenceTree; }

 private:
  //! Mappings to old reference indices (used when this object builds trees).
  std::vector<size_t> oldFromNewReferences;
  //! Reference tree.
  Tree* referenceTree;
  //! Reference set (data should be accessed using this).
  const MatType& referenceSet;

  //! If true, this object is responsible for deleting the trees.
  bool treeOwner;

  //! If true, O(n^2) naive computation is used.
  bool naive;
  //! If true, single-tree computation is used.
  bool singleMode;

  //! Instantiated distance metric.
  MetricType metric;
};

} // namespace range
} // namespace mlpack

// Include implementation.
#include "range_search_impl.hpp"

#endif
