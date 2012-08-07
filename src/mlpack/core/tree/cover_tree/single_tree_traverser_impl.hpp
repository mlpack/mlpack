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
  std::queue<double> pointScores;
  std::queue<double> baseCaseResults;

  pointQueue.push(&referenceNode);
  pointScores.push(0.0); // Cannot be pruned.

  // Evaluate first base case.
  baseCaseResults.push(rule.BaseCase(queryIndex, referenceNode.Point()));

  while (!pointQueue.empty())
  {
    CoverTree<MetricType, RootPointPolicy, StatisticType>* node =
        pointQueue.front();
    const double score = pointScores.front();
    const double baseCase = baseCaseResults.front();

    pointQueue.pop();
    pointScores.pop();
    baseCaseResults.pop();

    // First we (re)calculate the score of this node to find if we can prune it.
    double actualScore = rule.Rescore(queryIndex, *node, score);

    if (actualScore == DBL_MAX)
    {
      // Prune this node.
      ++numPrunes;
      continue; // Skip to next in queue.
    }

    // The base case is already evaluated.  So now we need to find out how to
    // recurse into the children.
    arma::vec baseCases(node->NumChildren());
    arma::vec childScores(node->NumChildren());

    // We already know the base case for the self child (that's the same as the
    // base case for us).
    baseCases[0] = baseCase;
    if (node->Child(0).NumChildren() == 0)
      childScores[0] = DBL_MAX; // Do not recurse into leaves (unnecessary).
    else
      childScores[0] = rule.Score(queryIndex, node->Child(0), baseCase);

    // Fill the rest of the children.
    for (size_t i = 1; i < node->NumChildren(); ++i)
    {
      baseCases[i] = rule.BaseCase(queryIndex, node->Child(i).Point());
      if (node->Child(i).NumChildren() == 0)
        childScores[i] = DBL_MAX; // Do not recurse into leaves (unnecessary).
      else
        childScores[i] = rule.Score(queryIndex, node->Child(i), baseCases[i]);
    }

    // Now sort by score.
    arma::uvec order = arma::sort_index(childScores);

    // Now add each to the queue.
    for (size_t i = 0; i < order.n_elem; ++i)
    {
      // Ensure we haven't hit the limit yet.
      const double childScore = childScores[order[i]];
      if (childScore == DBL_MAX)
      {
        numPrunes += (order.n_elem - i); // Prune the rest of the children.
        break; // Go on to next point.
      }

      pointQueue.push(&node->Child(order[i]));
      pointScores.push(childScore);
      baseCaseResults.push(baseCases[order[i]]);
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
