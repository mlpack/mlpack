/**
 * @file single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the single tree traverser for cover trees, which implements
 * a breadth-first traversal.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

#include <queue>

namespace mlpack {
namespace tree {

//! This is the structure the priority queue will use for traversal.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
struct CoverTreeQueueEntry
{
  //! The node this entry refers to.
  CoverTree<MetricType, RootPointPolicy, StatisticType>* node;
  //! The score of the node.
  double score;
  //! The index of the parent node.
  size_t parent;
  //! The base case evaluation (-1.0 if it has not been performed).
  double baseCase;

  //! Comparison operator.
  bool operator<(const CoverTreeQueueEntry& other) const
  {
    return (score < other.score);
  }
};



template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
CoverTree<MetricType, RootPointPolicy, StatisticType>::
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    CoverTree<MetricType, RootPointPolicy, StatisticType>& referenceNode)
{
  // This is a non-recursive implementation (which should be faster than a
  // recursive implementation).
  typedef CoverTreeQueueEntry<MetricType, RootPointPolicy, StatisticType>
      QueueType;

  // We will use this map as a priority queue.  Each key represents the scale,
  // and then the vector is all the nodes in that scale which need to be
  // investigated.  Because no point in a scale can add a point in its own
  // scale, we know that the vector for each scale is final when we get to it.
  // In addition, map is organized in such a way that rbegin() will return the
  // largest scale.
  std::map<int, std::vector<QueueType> > mapQueue;

  // Manually add the children of the first node.  These cannot be pruned
  // anyway.
  double rootBaseCase = rule.BaseCase(queryIndex, referenceNode.Point());
    
  // Create the score for the children.
  double rootChildScore = rule.Score(queryIndex, referenceNode, rootBaseCase);

  if (rootChildScore == DBL_MAX)
  {
    numPrunes += referenceNode.NumChildren();
  }
  else
  {
    // Don't add the self-leaf.
    size_t i = 0;
    if (referenceNode.Child(0).NumChildren() == 0)
    {
      ++numPrunes;
      i = 1;
    }

    for (/* i was set above. */; i < referenceNode.NumChildren(); ++i)
    {
      QueueType newFrame;
      newFrame.node = &referenceNode.Child(i);
      newFrame.score = rootChildScore;
      newFrame.baseCase = rootBaseCase;
      newFrame.parent = referenceNode.Point();

      // Put it into the map.
      mapQueue[newFrame.node->Scale()].push_back(newFrame);
    }
  }

  // Now begin the iteration through the map.
  typename std::map<int, std::vector<QueueType> >::reverse_iterator rit =
      mapQueue.rbegin();

  // We will treat the leaves differently (below).
  while ((*rit).first != INT_MIN)
  {
    // Get a reference to the current scale.
    std::vector<QueueType>& scaleVector = (*rit).second;

    // Before beginning all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    // Now loop over each element.
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      // Get a reference to the current element.
      const QueueType& frame = scaleVector.at(i);

      CoverTree<MetricType, RootPointPolicy, StatisticType>* node = frame.node;
      const double score = frame.score;
      const size_t parent = frame.parent;
      const size_t point = node->Point();
      double baseCase = frame.baseCase;

      // First we recalculate the score of this node to find if we can prune it.
      if (rule.Rescore(queryIndex, *node, score) == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      // If we are a self-child, the base case has already been evaluated.
      if (point != parent)
        baseCase = rule.BaseCase(queryIndex, point);

      // Create the score for the children.
      const double childScore = rule.Score(queryIndex, *node, baseCase);

      // Now if this childScore is DBL_MAX we can prune all children.  In this
      // recursion setup pruning is all or nothing for children.
      if (childScore == DBL_MAX)
      {
        numPrunes += node->NumChildren();
        continue;
      }

      // Don't add the self-leaf.
      size_t j = 0;
      if (node->Child(0).NumChildren() == 0)
      {
        ++numPrunes;
        j = 1;
      }

      for (/* j is already set. */; j < node->NumChildren(); ++j)
      {
        QueueType newFrame;
        newFrame.node = &node->Child(j);
        newFrame.score = childScore;
        newFrame.baseCase = baseCase;
        newFrame.parent = point;

        mapQueue[newFrame.node->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    mapQueue.erase((*rit).first);
  }

  // Now deal with the leaves.
  for (size_t i = 0; i < mapQueue[INT_MIN].size(); ++i)
  {
    const QueueType& frame = mapQueue[INT_MIN].at(i);

    CoverTree<MetricType, RootPointPolicy, StatisticType>* node = frame.node;
    const double score = frame.score;
    const size_t point = node->Point();

    // First, recalculate the score of this node to find if we can prune it.
    double actualScore = rule.Rescore(queryIndex, *node, score);

    if (actualScore == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }

    // There are no self-leaves; evaluate the base case.
    const double baseCase = rule.BaseCase(queryIndex, point);
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
