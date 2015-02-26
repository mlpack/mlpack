/**
 * @file breadth_first_dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the BreadthFirstDualTreeTraverser for BinarySpaceTree.
 * This is a way to perform a dual-tree traversal of two trees.  The trees must
 * be the same type.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "breadth_first_dual_tree_traverser.hpp"

#include <queue>

namespace mlpack {
namespace tree {

template<typename BoundType,
         typename StatisticType,
         typename MatType,
         typename SplitType>
template<typename RuleType>
BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>::
BreadthFirstDualTreeTraverser<RuleType>::BreadthFirstDualTreeTraverser(
    RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do. */ }

template<typename TreeType, typename TraversalInfoType>
struct QueueFrame
{
  TreeType* queryNode;
  TreeType* referenceNode;
  size_t queryDepth;
  double score;
  TraversalInfoType traversalInfo;
};

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

template<typename BoundType,
         typename StatisticType,
         typename MatType,
         typename SplitType>
template<typename RuleType>
void BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>::
BreadthFirstDualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>& queryRoot,
    BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>&
        referenceRoot)
{
  // Increment the visit counter.
  ++numVisited;

  // Store the current traversal info.
  traversalInfo = rule.TraversalInfo();

  typedef BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>
      TreeType;

  // Must score the root combination.
  const double rootScore = rule.Score(queryRoot, referenceRoot);
  if (rootScore == DBL_MAX)
    return; // This probably means something is wrong.

  typedef QueueFrame<TreeType, typename RuleType::TraversalInfoType>
      QueueFrameType;
  std::priority_queue<QueueFrameType> queue;

  QueueFrameType rootFrame;
  rootFrame.queryNode = &queryRoot;
  rootFrame.referenceNode = &referenceRoot;
  rootFrame.queryDepth = 0;
  rootFrame.score = 0.0;
  rootFrame.traversalInfo = rule.TraversalInfo();

  queue.push(rootFrame);

  while (!queue.empty())
  {
    QueueFrameType currentFrame = queue.top();
    queue.pop();

    TreeType& queryNode = *currentFrame.queryNode;
    TreeType& referenceNode = *currentFrame.referenceNode;
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
      for (size_t query = queryNode.Begin(); query < queryNode.End(); ++query)
      {
        // See if we need to investigate this point (this function should be
        // implemented for the single-tree recursion too).  Restore the
        // traversal information first.
//        const double childScore = rule.Score(query, referenceNode);

//        if (childScore == DBL_MAX)
//          continue; // We can't improve this particular point.

        for (size_t ref = referenceNode.Begin(); ref < referenceNode.End();
            ++ref)
          rule.BaseCase(query, ref);

        numBaseCases += referenceNode.Count();
      }
    }
    else if ((!queryNode.IsLeaf()) && referenceNode.IsLeaf())
    {
      // We have to recurse down the query node.
      QueueFrameType fl = { queryNode.Left(), &referenceNode, queryDepth + 1,
          score, rule.TraversalInfo() };
      queue.push(fl);

      QueueFrameType fr = { queryNode.Right(), &referenceNode, queryDepth + 1,
          score, ti };
      queue.push(fr);
    }
    else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
    {
      // We have to recurse down the reference node.  In this case the recursion
      // order does matter.  Before recursing, though, we have to set the
      // traversal information correctly.
      QueueFrameType fl = { &queryNode, referenceNode.Left(), queryDepth,
          score, rule.TraversalInfo() };
      queue.push(fl);

      QueueFrameType fr = { &queryNode, referenceNode.Right(), queryDepth,
          score, ti };
      queue.push(fr);
    }
    else
    {
      // We have to recurse down both query and reference nodes.  Because the
      // query descent order does not matter, we will go to the left query child
      // first.  Before recursing, we have to set the traversal information
      // correctly.
      QueueFrameType fll = { queryNode.Left(), referenceNode.Left(),
          queryDepth + 1, score, rule.TraversalInfo() };
      queue.push(fll);

      QueueFrameType flr = { queryNode.Left(), referenceNode.Right(),
          queryDepth + 1, score, rule.TraversalInfo() };
      queue.push(flr);

      QueueFrameType frl = { queryNode.Right(), referenceNode.Left(),
          queryDepth + 1, score, rule.TraversalInfo() };
      queue.push(frl);

      QueueFrameType frr = { queryNode.Right(), referenceNode.Right(),
          queryDepth + 1, score, rule.TraversalInfo() };
      queue.push(frr);
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP
