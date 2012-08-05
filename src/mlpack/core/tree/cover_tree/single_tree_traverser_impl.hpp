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
  std::queue<CoverTree<MetricType, RootPointPolicy, StatisticType>*> pointQueue;
  std::queue<size_t> parentPoints; // For if this tree has self-children.

  pointQueue.push(&referenceNode);
  parentPoints.push(size_t() - 1); // Invalid value.

  while (!pointQueue.empty())
  {
    CoverTree<MetricType, RootPointPolicy, StatisticType>* node =
        pointQueue.front();
    pointQueue.pop();

    // Check if we can prune this node.
    if (rule.CanPrune(queryIndex, *node))
    {
      parentPoints.pop(); // Pop the parent point off.

      ++numPrunes;
      continue;
    }

    // If this tree type has self-children, we need to make sure we don't run
    // the base case if the parent already had it run.
    size_t baseCaseStart = 0;
    if (parentPoints.front() == node->Point(0))
      baseCaseStart = 1; // Skip base case we've already evaluated.

    parentPoints.pop();

    // First run the base case for any points this node might hold.
    for (size_t i = baseCaseStart; i < node->NumPoints(); ++i)
      rule.BaseCase(queryIndex, node->Point(i));

    // Now push children (and their parent points) into the FIFO.  Maybe it
    // would be better to push these back in a particular order.
    const size_t parentPoint = node->Point(0);
    for (size_t i = 0; i < node->NumChildren(); ++i)
    {
      pointQueue.push(&(node->Child(i)));
      parentPoints.push(parentPoint);
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
