/**
 * @file core/tree/cover_tree/dual_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>
#include <queue>
#include "recursion_sets.hpp"

namespace mlpack {

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
class CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    DualTreeTraverser
{
 public:
  /**
   * Initialize the dual tree traverser with the given rule type.
   */
  DualTreeTraverser(RuleType& rule);

  /**
   * Traverse the two specified trees.
   *
   * @param queryNode Root of query tree.
   * @param referenceNode Root of reference tree.
   */
  void Traverse(CoverTree& queryNode, CoverTree& referenceNode);

  //! Get the number of pruned nodes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of pruned nodes.
  size_t& NumPrunes() { return numPrunes; }

  ///// These are all fake because this is a patch for kd-trees only and I still
  ///// want it to compile!
  size_t NumVisited() const { return 0; }
  size_t NumScores() const { return 0; }
  size_t NumBaseCases() const { return 0; }

 private:
  //! The instantiated rule set for pruning branches.
  RuleType& rule;

  //! The number of pruned nodes.
  size_t numPrunes;

  // Struct used for traversal.
  typedef DualCoverTreeMapEntry<CoverTree, RuleType> MapEntryType;

  // Prepare map for recursion.
  void PruneMap(
      CoverTree& queryNode,
      CoverTreeRecursionSets<MapEntryType, 8>& recursionSets,
      CoverTreeRecursionSets<MapEntryType, 8>& childRecursionSets);

  void ReferenceRecursion(
      CoverTree& queryNode,
      CoverTreeRecursionSets<MapEntryType, 8>& recursionSets);
};

} // namespace mlpack

// Include implementation.
#include "dual_tree_traverser_impl.hpp"

#endif
