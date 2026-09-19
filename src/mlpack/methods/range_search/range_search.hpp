/**
 * @file methods/range_search/range_search.hpp
 * @author Ryan Curtin
 *
 * Defines the RangeSearch class, which performs a generalized range search on
 * points.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include "range_search_stat.hpp"

namespace mlpack {

//! Forward declaration.
template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
class LeafSizeRSWrapper;

// RangeSearchStrategy represents the different range search strategies
// available.
enum RangeSearchStrategy
{
  NAIVE,
  SINGLE_TREE,
  DUAL_TREE
};

/**
 * The RangeSearch class is a template class for performing range searches.  It
 * is implemented in the style of a generalized tree-independent dual-tree
 * algorithm; for more details on the actual algorithm, see the RangeSearchRules
 * class.
 *
 * @tparam DistanceType Metric to use for range search calculations.
 * @tparam MatType Type of data to use.
 * @tparam TreeType Type of tree to use; must satisfy the TreeType policy API.
 */
template<typename DistanceType = EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = KDTree>
class RangeSearch
{
 public:
  //! Convenience typedef.
  using Tree = TreeType<DistanceType, RangeSearchStat, MatType>;
  //! The type of Matrix.
  using Mat = MatType;
  //! The type of element held in MatType.
  using ElemType = typename MatType::elem_type;

  /**
   * Initialize the RangeSearch object with a given reference dataset (this is
   * the dataset which is searched).  Optionally, perform the computation in
   * naive mode or single-tree mode depending on the setting of `strategy`.
   * Additionally, an instantiated distance metric can be given, for cases where
   * the distance metric holds data.
   *
   * This method will move the matrices to internal copies, which are
   * rearranged during tree-building.  You can avoid creating an extra copy by
   * pre-constructing the trees and passing them in using std::move.
   *
   * @param referenceSet Reference dataset.
   * @param strategy Range search strategy.
   * @param distance Instantiated distance metric.
   */
  RangeSearch(MatType referenceSet,
              const RangeSearchStrategy strategy = DUAL_TREE,
              const DistanceType distance = DistanceType());

  [[deprecated("Will be removed in mlpack 5.0.0.  Instead of a boolean for "
               "'naive' and 'singleMode', pass a RangeSearchStrategy.")]]
  RangeSearch(MatType referenceSet,
              const bool naive,
              const bool singleMode,
              const DistanceType distance = DistanceType());

  /**
   * Initialize the RangeSearch object with the given pre-constructed reference
   * tree (this is the tree built on the reference set, which is the set that is
   * searched).  Optionally, choose to use single-tree mode, which will not
   * build a tree on query points by specifying `strategy` as `SINGLE_TREE`.
   * Additionally, an instantiated distance metric can be given, for cases where
   * the distance metric holds data.
   *
   * There is no copying of the data matrices in this constructor (because
   * tree-building is not necessary), so this is the constructor to use when
   * copies absolutely must be avoided.
   *
   * @note
   * Because tree-building (at least with BinarySpaceTree) modifies the ordering
   * of a matrix, be aware that mapping of the points back to their original
   * indices is not done when this constructor is used.
   *
   * @param referenceTree Pre-built tree for reference points.
   * @param strategy Range search strategy.
   * @param distance Instantiated distance metric.
   */
  RangeSearch(Tree referenceTree,
              const RangeSearchStrategy = DUAL_TREE,
              const DistanceType distance = DistanceType());

  [[deprecated("Will be removed in mlpack 5.0.0.  Instead of a boolean for "
               "'singleMode', pass a RangeSearchStrategy.")]]
  RangeSearch(Tree* referenceTree,
              const bool singleMode,
              const DistanceType distance = DistanceType());

  /**
   * Initialize the RangeSearch object without any reference data.  If the
   * monochromatic Search() is called before a reference set is set with
   * Train(), no results will be returned (since the reference set is empty).
   *
   * @param strategy Range search strategy.
   * @param distance Instantiated distance metric.
   */
  RangeSearch(const RangeSearchStrategy strategy = DUAL_TREE,
              const DistanceType distance = DistanceType());

  RangeSearch(const bool naive,
              const bool singleMode,
              const DistanceType distance = DistanceType());

  /**
   * Construct the RangeSearch model as a copy of the given model.  Note that
   * this may be computationally intensive!
   *
   * @param other RangeSearch model to copy.
   */
  RangeSearch(const RangeSearch& other);

  /**
   * Construct the RangeSearch model by taking ownership of the given model.
   *
   * @param other RangeSearch model to take ownership of.
   */
  RangeSearch(RangeSearch&& other);

  /**
   * Deep copy the given RangeSearch model.
   * 
   * @param other RangeSearch model to copy.
   */
  RangeSearch& operator=(const RangeSearch& other);

  /**
   * Move the given RangeSearch model.
   *
   * @param other RangeSearch model to move.
   */
  RangeSearch& operator=(RangeSearch&& other);

  /**
   * Destroy the RangeSearch object.  If trees were created, they will be
   * deleted.
   */
  ~RangeSearch();

  /**
   * Set the reference set to a new reference set, and build a tree if
   * necessary.  This method is called 'Train()' in order to match the rest of
   * the mlpack abstractions, even though calling this "training" is maybe a
   * bit of a stretch.
   *
   * Use std::move to pass in the reference set if the old copy is no longer
   * needed.
   *
   * @param referenceSet New set of reference data.
   */
  void Train(MatType referenceSet);

  /**
   * Set the reference tree to a new reference tree.
   */
  void Train(Tree referenceTree);

  [[deprecated("Will be removed in mlpack 5.0.0.  Pass a direct reference to "
               "the tree instead (possibly with std::move() to transfer "
               "ownership).")]]
  void Train(Tree* referenceTree);

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
              const RangeType<ElemType>& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<ElemType>>& distances);

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
  void Search(Tree& queryTree,
              const RangeType<ElemType>& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<ElemType>>& distances);

  [[deprecated("Will be removed in mlpack 5.0.0.  Pass a reference to the query"
               " tree instead.")]]
  void Search(Tree* queryTree,
              const RangeType<ElemType>& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<ElemType>>& distances);

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
   * @param range Range of distances in which to search.
   * @param neighbors Object which will hold the list of neighbors for each
   *      point which fell into the given range, for each query point.
   * @param distances Object which will hold the list of distances for each
   *      point which fell into the given range, for each query point.
   */
  void Search(const RangeType<ElemType>& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<ElemType>>& distances);

  // Get the search mode that will be used.
  const RangeSearchStrategy& Strategy() const { return strategy; }
  // Modify the range search strategy that will be used.
  RangeSearchStrategy& Strategy() { return strategy; }

  //! Get whether single-tree search is being used.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use Strategy() instead.")]]
  bool SingleMode() const { return singleMode; }
  //! Modify whether single-tree search is being used.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use Strategy() instead.")]]
  bool& SingleMode() { return singleMode; }

  //! Get whether naive search is being used.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use Strategy() instead.")]]
  bool Naive() const { return naive; }
  //! Modify whether naive search is being used.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use Strategy() instead.")]]
  bool& Naive() { return naive; }

  //! Get the number of base cases during the last search.
  size_t BaseCases() const { return baseCases; }
  //! Get the number of scores during the last search.
  size_t Scores() const { return scores; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

  //! Return the reference set.
  const MatType& ReferenceSet() const { return *referenceSet; }

  //! Return the reference tree (or NULL if in naive mode).
  Tree* ReferenceTree() { return referenceTree; }

 private:
  //! Mappings to old reference indices (used when this object builds trees).
  std::vector<size_t> oldFromNewReferences;
  //! Reference tree.
  Tree* referenceTree;
  //! Reference set (data should be accessed using this).  In some situations we
  //! may be the owner of this.
  const MatType* referenceSet;

  //! If true, this object is responsible for deleting the trees.
  bool treeOwner;

  RangeSearchStrategy strategy;

  //! If true, O(n^2) naive computation is used.
  bool naive;
  //! If true, single-tree computation is used.
  bool singleMode;

  //! Instantiated distance metric.
  DistanceType distance;

  //! The total number of base cases during the last search.
  size_t baseCases;
  //! The total number of scores during the last search.
  size_t scores;

  //! For access to mappings when building models.
  friend class LeafSizeRSWrapper<TreeType>;
};

} // namespace mlpack

// The CEREAL_TEMPLATE_CLASS_VERSION() macro does not work with template
// template parameters so we write it manually.
namespace cereal {
namespace detail {

template<typename DistanceType = EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = KDTree>
struct Version<mlpack::RangeSearch<DistanceType, MatType, TreeType>
{
  static std::uint32_t registerVersion()
  {
    ::cereal::detail::StaticObject<Versions>::getInstance().mapping.emplace(
        std::type_index(typeid(mlpack::RangeSearch<DistanceType, MatType,
        TreeType>)).hash_code(), 1);
    return 1;
  }

  static inline const std::uint32_t version = registerVersion();

  static void unused() { (void) version; }
}; /* end Version */

} // namespace detail
} // namespace cereal

// Include implementation.
#include "range_search_impl.hpp"

#endif
