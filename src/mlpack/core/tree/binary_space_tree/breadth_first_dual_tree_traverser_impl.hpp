/**
 * @file breadth_first_dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the BreadthFirstDualTreeTraverser for BinarySpaceTree.
 * This is a way to perform a dual-tree traversal of two trees.  The trees must
 * be the same type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "breadth_first_dual_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
BreadthFirstDualTreeTraverser<RuleType>::BreadthFirstDualTreeTraverser(
    RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do. */ }

template<typename TreeType, typename TraversalInfoType>
bool operator<(const QueueFrame<TreeType, TraversalInfoType>& a,
               const QueueFrame<TreeType, TraversalInfoType>& b)
{
  if (a.queryDepth > b.queryDepth)
    return true;
  else if ((a.queryDepth == b.queryDepth) && (a.score > b.score))
    return true;
  return false;
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
void BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
BreadthFirstDualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        queryRoot,
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        referenceRoot)
{
  // Increment the visit counter.
  ++numVisited;

  // Store the current traversal info.
  traversalInfo = rule.TraversalInfo();

  // Must score the root combination.
  const double rootScore = rule.Score(queryRoot, referenceRoot);
  if (rootScore == DBL_MAX)
    return; // This probably means something is wrong.

  std::priority_queue<QueueFrameType> queue;

  QueueFrameType rootFrame;
  rootFrame.queryNode = &queryRoot;
  rootFrame.referenceNode = &referenceRoot;
  rootFrame.queryDepth = 0;
  rootFrame.score = 0.0;
  rootFrame.traversalInfo = rule.TraversalInfo();

  queue.push(rootFrame);

  // Start the traversal.
  Traverse(queryRoot, queue);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
void BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
BreadthFirstDualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        queryNode,
    std::priority_queue<QueueFrameType>& referenceQueue)
{
  // Store queues for the children.  We will recurse into the children once our
  // queue is empty.
  std::priority_queue<QueueFrameType> leftChildQueue;
  std::priority_queue<QueueFrameType> rightChildQueue;

  while (!referenceQueue.empty())
  {
    QueueFrameType currentFrame = referenceQueue.top();
    referenceQueue.pop();

    BinarySpaceTree& queryNode = *currentFrame.queryNode;
    BinarySpaceTree& referenceNode = *currentFrame.referenceNode;
    typename RuleType::TraversalInfoType ti = currentFrame.traversalInfo;
    rule.TraversalInfo() = ti;
    const size_t queryDepth = currentFrame.queryDepth;

    double score = rule.Score(queryNode, referenceNode);
    ++numScores;

    if (score == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }

    // If both are leaves, we must evaluate the base case.
    if (queryNode.IsLeaf() && referenceNode.IsLeaf())
    {
      // Loop through each of the points in each node.
      const size_t queryEnd = queryNode.Begin() + queryNode.Count();
      const size_t refEnd = referenceNode.Begin() + referenceNode.Count();
      for (size_t query = queryNode.Begin(); query < queryEnd; ++query)
      {
        // See if we need to investigate this point (this function should be
        // implemented for the single-tree recursion too).  Restore the
        // traversal information first.
//        const double childScore = rule.Score(query, referenceNode);

//        if (childScore == DBL_MAX)
//          continue; // We can't improve this particular point.

        for (size_t ref = referenceNode.Begin(); ref < refEnd; ++ref)
          rule.BaseCase(query, ref);

        numBaseCases += referenceNode.Count();
      }
    }
    else if ((!queryNode.IsLeaf()) && referenceNode.IsLeaf())
    {
      // We have to recurse down the query node.
      QueueFrameType fl = { queryNode.Left(), &referenceNode, queryDepth + 1,
          score, rule.TraversalInfo() };
      leftChildQueue.push(fl);

      QueueFrameType fr = { queryNode.Right(), &referenceNode, queryDepth + 1,
          score, ti };
      rightChildQueue.push(fr);
    }
    else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
    {
      // We have to recurse down the reference node.  In this case the recursion
      // order does matter.  Before recursing, though, we have to set the
      // traversal information correctly.
      QueueFrameType fl = { &queryNode, referenceNode.Left(), queryDepth,
          score, rule.TraversalInfo() };
      referenceQueue.push(fl);

      QueueFrameType fr = { &queryNode, referenceNode.Right(), queryDepth,
          score, ti };
      referenceQueue.push(fr);
    }
    else
    {
      // We have to recurse down both query and reference nodes.  Because the
      // query descent order does not matter, we will go to the left query child
      // first.  Before recursing, we have to set the traversal information
      // correctly.
      QueueFrameType fll = { queryNode.Left(), referenceNode.Left(),
          queryDepth + 1, score, rule.TraversalInfo() };
      leftChildQueue.push(fll);

      QueueFrameType flr = { queryNode.Left(), referenceNode.Right(),
          queryDepth + 1, score, rule.TraversalInfo() };
      leftChildQueue.push(flr);

      QueueFrameType frl = { queryNode.Right(), referenceNode.Left(),
          queryDepth + 1, score, rule.TraversalInfo() };
      rightChildQueue.push(frl);

      QueueFrameType frr = { queryNode.Right(), referenceNode.Right(),
          queryDepth + 1, score, rule.TraversalInfo() };
      rightChildQueue.push(frr);
    }
  }

  // Now, recurse into the left and right children queues.  The order doesn't
  // matter.
  if (leftChildQueue.size() > 0)
    Traverse(*queryNode.Left(), leftChildQueue);
  if (rightChildQueue.size() > 0)
    Traverse(*queryNode.Right(), rightChildQueue);
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP
