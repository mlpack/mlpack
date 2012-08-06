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
  std::queue<double> pointScores;

  pointQueue.push(&referenceNode);
  parentPoints.push(size_t() - 1); // Invalid value.
  pointScores.push(0.0); // Cannot be pruned.

  while (!pointQueue.empty())
  {
    CoverTree<MetricType, RootPointPolicy, StatisticType>* node =
        pointQueue.front();
    const size_t parent = parentPoints.front();
    const double score = pointScores.front();
    const size_t point = node->Point(); // The point held by this node.

    pointQueue.pop();
    parentPoints.pop();
    pointScores.pop();

    // See if this point should still be recursed into.
    if (rule.Rescore(queryIndex, *node, score) == DBL_MAX)
    {
      ++numPrunes;
      continue; // Pruned!
    }

    // Evaluate the base case, but only if this node is not holding the same
    // point as its parent.
    if (parent != point)
      rule.BaseCase(queryIndex, point);

    // Now get the scores for recursion.
    arma::vec scores(node->NumChildren());

    for (size_t i = 0; i < node->NumChildren(); ++i)
      scores[i] = rule.Score(queryIndex, node->Child(i));

    // Now sort by distance (smallest first).
    arma::uvec order = arma::sort_index(scores);

    for (size_t i = 0; i < order.n_elem; ++i)
    {
      // Ensure we haven't hit the limit yet.
      const double childScore = scores[order[i]];
      if (childScore == DBL_MAX)
      {
        numPrunes += (order.n_elem - i); // Prune the rest of the children.
        break; // Go on to next point.
      }

      pointQueue.push(&node->Child(order[i]));
      parentPoints.push(point);
      pointScores.push(childScore);
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
