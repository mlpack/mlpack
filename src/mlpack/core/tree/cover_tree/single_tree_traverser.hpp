/**
 * @file core/tree/cover_tree/single_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the SingleTreeTraverser for the cover tree.  This implements a
 * single-tree breadth-first recursion with a pruning rule and a base case (two
 * point) rule.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>

#include "cover_tree.hpp"

namespace mlpack {

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
class CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    SingleTreeTraverser
{
 public:
  /**
   * Initialize the single tree traverser with the given rule.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is used as
   *      the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, CoverTree& referenceNode);

  //! Get the number of prunes so far.
  size_t NumPrunes() const { return numPrunes; }
  //! Set the number of prunes (good for a reset to 0).
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  //! The number of nodes which have been pruned during traversal.
  size_t numPrunes;
};

} // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif
