/**
 * @file breadth_first_dual_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the BreadthFirstDualTreeTraverser for the BinarySpaceTree tree type.
 * This is a nested class of BinarySpaceTree which traverses two trees in a
 * breadth-first manner with a given set of rules which indicate the branches
 * which can be pruned and the order in which to recurse.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>
#include <queue>

#include "../binary_space_tree.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType, typename TraversalInfoType>
struct QueueFrame
{
  TreeType* queryNode;
  TreeType* referenceNode;
  size_t queryDepth;
  double score;
  TraversalInfoType traversalInfo;
};

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
class BinarySpaceTree<MetricType, StatisticType, MatType, BoundType,
                      SplitType>::BreadthFirstDualTreeTraverser
{
 public:
  /**
   * Instantiate the dual-tree traverser with the given rule set.
   */
  BreadthFirstDualTreeTraverser(RuleType& rule);

  typedef QueueFrame<BinarySpaceTree, typename RuleType::TraversalInfoType>
      QueueFrameType;

  /**
   * Traverse the two trees.  This does not reset the number of prunes.
   *
   * @param queryNode The query node to be traversed.
   * @param referenceNode The reference node to be traversed.
   * @param score The score of the current node combination.
   */
  void Traverse(BinarySpaceTree& queryNode,
                BinarySpaceTree& referenceNode);
  void Traverse(BinarySpaceTree& queryNode,
                std::priority_queue<QueueFrameType>& referenceQueue);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of prunes.
  size_t& NumPrunes() { return numPrunes; }

  //! Get the number of visited combinations.
  size_t NumVisited() const { return numVisited; }
  //! Modify the number of visited combinations.
  size_t& NumVisited() { return numVisited; }

  //! Get the number of times a node combination was scored.
  size_t NumScores() const { return numScores; }
  //! Modify the number of times a node combination was scored.
  size_t& NumScores() { return numScores; }

  //! Get the number of times a base case was calculated.
  size_t NumBaseCases() const { return numBaseCases; }
  //! Modify the number of times a base case was calculated.
  size_t& NumBaseCases() { return numBaseCases; }

 private:
  //! Reference to the rules with which the trees will be traversed.
  RuleType& rule;

  //! The number of prunes.
  size_t numPrunes;

  //! The number of node combinations that have been visited during traversal.
  size_t numVisited;

  //! The number of times a node combination was scored.
  size_t numScores;

  //! The number of times a base case was calculated.
  size_t numBaseCases;

  //! Traversal information, held in the class so that it isn't continually
  //! being reallocated.
  typename RuleType::TraversalInfoType traversalInfo;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "breadth_first_dual_tree_traverser_impl.hpp"

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_HPP

